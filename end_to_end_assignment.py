"""
15-Minute Time Slot Assignment Algorithm with Contiguous Assignment Preference

This algorithm assigns participants to 15-minute time slots based on their hour-long 
preferences. Each participant gets exactly ONE 15-minute slot.

Key Features:
- Assigns exactly 1 participant per 15-minute slot
- Each participant gets exactly 1 slot (15 minutes only)
- Expands hour preferences to 15-minute slots
- Maximizes preference satisfaction (primary objective)
- Prefers grouping participants in consecutive slots (secondary objective)
- Uses linear programming for optimal solutions

Contiguity Preference:
The algorithm prefers to group participants in consecutive 15-minute slots
to create blocks of meetings. For example, with 6 participants, it will prefer:
- 4 consecutive slots on Monday morning (different participants)
Rather than:
- 6 scattered slots throughout the week

This makes scheduling more efficient by reducing gaps between meetings.
"""

import pandas as pd
import numpy as np
from pulp import *
import openpyxl
import re
from datetime import datetime, timedelta
import argparse

def load_and_process_data(file_path):
    """
    Load the spreadsheet and process preference data.
    
    Args:
        file_path: Path to the Excel/CSV file
    
    Returns:
        df: DataFrame with participant data
        hour_slots: List of hour-long time slot column names (C through BF)
        participants: List of participant names/IDs
    """
    # Load the data
    df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
    
    # Get time slot columns (C through BF)
    # Excel columns C through BF correspond to indices 2 through 57
    hour_slot_cols = df.columns[2:58]  # Adjust if your column range is different
    hour_slots = list(hour_slot_cols)
    
    # Get participant identifiers (assuming first column is name/ID)
    participants = df.iloc[:, 1].tolist()
    
    return df, hour_slots, participants

def expand_to_15min_slots(hour_slots):
    """
    Expand hour-long slots to 15-minute slots.
    Each hour slot becomes 4 fifteen-minute slots.
    
    Returns:
        fifteen_min_slots: List of 15-minute slot names
        hour_to_15min_map: Dict mapping hour slot index to 15-min slot indices
    """
    fifteen_min_slots = []
    hour_to_15min_map = {}
    
    for hour_idx, hour_slot in enumerate(hour_slots):
        hour_to_15min_map[hour_idx] = []
        
        # Parse the hour slot to get time info
        # Try to extract time and day information
        slot_lower = hour_slot.lower()
        
        # Find time pattern
        time_pattern = r'(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.|a|p)'
        time_match = re.search(time_pattern, slot_lower)
        
        if time_match:
            # Create 4 fifteen-minute slots
            offsets = [0, 15, 30, 45]
            for offset in offsets:
                # Create slot name with offset
                if "[" in hour_slot:
                    # Handle format like "Quiz 1 Time Preference [Mon 8:30p]"
                    base = hour_slot.split('[')[0].strip()
                    bracket_part = hour_slot[hour_slot.index('['):hour_slot.index(']')+1]
                    
                    # Extract day and time from bracket
                    bracket_content = bracket_part[1:-1]  # Remove brackets
                    parts = bracket_content.split()
                    if len(parts) >= 2:
                        day = parts[0]
                        time_str = parts[1]
                        
                        # Parse time
                        hour = int(time_match.group(1))
                        minute = int(time_match.group(2))
                        period = time_match.group(3)
                        
                        # Calculate new time
                        total_minutes = hour * 60 + minute + offset
                        new_hour = total_minutes // 60
                        new_minute = total_minutes % 60
                        
                        # Format new time
                        if period.startswith('p') and new_hour < 12:
                            new_hour += 12
                        
                        # Handle overflow to next hour
                        if new_hour >= 13 and period.startswith('p'):
                            display_hour = new_hour - 12
                        elif new_hour > 12:
                            display_hour = new_hour - 12
                        elif new_hour == 0:
                            display_hour = 12
                        else:
                            display_hour = new_hour
                            
                        # Keep the same am/pm format style
                        if len(period) == 1:  # Just 'p' or 'a'
                            period_str = period
                        else:
                            period_str = period
                            
                        # Create 15-minute slot name
                        start_time = f"{display_hour}:{new_minute:02d}{period_str}"
                        
                        # Calculate end time
                        end_minutes = total_minutes + 15
                        end_hour = end_minutes // 60
                        end_minute = end_minutes % 60
                        
                        if period.startswith('p') and end_hour < 12:
                            end_hour += 12
                            
                        if end_hour >= 13 and period.startswith('p'):
                            display_end_hour = end_hour - 12
                        elif end_hour > 12:
                            display_end_hour = end_hour - 12
                        elif end_hour == 0:
                            display_end_hour = 12
                        else:
                            display_end_hour = end_hour
                            
                        end_time = f"{display_end_hour}:{end_minute:02d}{period_str}"
                        
                        slot_name = f"{base} [{day} {start_time}-{end_time}]"
                    else:
                        # Fallback format
                        slot_name = f"{hour_slot} (+{offset:02d}min)"
                else:
                    # Standard format - add offset
                    slot_name = f"{hour_slot} (+{offset:02d}min)"
                
                fifteen_min_slots.append(slot_name)
                hour_to_15min_map[hour_idx].append(len(fifteen_min_slots) - 1)
        else:
            # If we can't parse the time, just append offset
            for offset in [0, 15, 30, 45]:
                slot_name = f"{hour_slot} (+{offset:02d}min)"
                fifteen_min_slots.append(slot_name)
                hour_to_15min_map[hour_idx].append(len(fifteen_min_slots) - 1)
    
    return fifteen_min_slots, hour_to_15min_map

def create_expanded_preference_matrix(df, hour_slots, participants, hour_to_15min_map, n_15min_slots):
    """
    Create preference and availability matrices for 15-minute slots based on hour preferences.
    
    Returns:
        preference_matrix: 2D numpy array of preferences for 15-min slots
        availability_matrix: Binary matrix (1 if available, 0 if not) for 15-min slots
    """
    n_participants = len(participants)
    
    preference_matrix = np.zeros((n_participants, n_15min_slots))
    availability_matrix = np.zeros((n_participants, n_15min_slots))
    
    for i, participant in enumerate(participants):
        for hour_idx, hour_slot in enumerate(hour_slots):
            pref = df.iloc[i][hour_slot]
            
            # Determine preference value
            if pref == "This is a great time for me!":
                pref_value = 2
                avail_value = 1
            elif pref == "I can make it at this time.":
                pref_value = 1
                avail_value = 1
            else:  # "I cannot make this time."
                pref_value = 0
                avail_value = 0
            
            # Apply to all 15-minute slots within this hour
            for slot_15min_idx in hour_to_15min_map[hour_idx]:
                preference_matrix[i, slot_15min_idx] = pref_value
                availability_matrix[i, slot_15min_idx] = avail_value
    
    return preference_matrix, availability_matrix

def identify_consecutive_15min_slots(fifteen_min_slots):
    """
    Identify which 15-minute slots are consecutive.
    
    Returns:
        adjacency_matrix: n_slots x n_slots matrix where adjacency_matrix[i][j] = 1 
                         if slot j immediately follows slot i (15 minutes later)
    """
    n_slots = len(fifteen_min_slots)
    adjacency_matrix = np.zeros((n_slots, n_slots))
    
    # Simple approach: slots are consecutive if they're adjacent in our list
    # and belong to the same day
    for i in range(n_slots - 1):
        # Check if slot i+1 follows slot i
        slot_i = fifteen_min_slots[i]
        slot_next = fifteen_min_slots[i + 1]
        
        # Extract day information to ensure same day
        day_i = None
        day_next = None
        
        # Try to extract day
        if '[' in slot_i and ']' in slot_i:
            bracket_i = slot_i[slot_i.index('['):slot_i.index(']')+1]
            parts_i = bracket_i[1:-1].split()
            if parts_i:
                day_i = parts_i[0]
                
        if '[' in slot_next and ']' in slot_next:
            bracket_next = slot_next[slot_next.index('['):slot_next.index(']')+1]
            parts_next = bracket_next[1:-1].split()
            if parts_next:
                day_next = parts_next[0]
        
        # If same day and consecutive in our ordering, they're adjacent
        if day_i and day_next and day_i == day_next:
            # Check if they're in the same hour block or consecutive 15-min slots
            if i % 4 < 3:  # Not the last slot of an hour
                adjacency_matrix[i][i + 1] = 1
            elif i + 1 < n_slots:  # Check if next hour starts immediately after
                # This handles hour boundaries (e.g., 9:45-10:00 followed by 10:00-10:15)
                adjacency_matrix[i][i + 1] = 1
    
    return adjacency_matrix

def solve_assignment(participants, fifteen_min_slots, preference_matrix, availability_matrix, contiguity_bonus=0.1):
    """
    Solve the assignment problem using linear programming.
    Each participant gets exactly one 15-minute slot.
    Each 15-minute slot can have at most 1 participant.
    
    Args:
        participants: List of participant names
        fifteen_min_slots: List of 15-minute slot names
        preference_matrix: Matrix of preferences for 15-min slots
        availability_matrix: Binary matrix of availability for 15-min slots
        contiguity_bonus: Bonus weight for using consecutive slots (0-1, default 0.1)
                         Encourages grouping participants in consecutive time slots
    
    Returns:
        assignments: Dictionary mapping participants to their assigned 15-minute slots
        unassigned: List of participants who couldn't be assigned
    """
    n_participants = len(participants)
    n_slots = len(fifteen_min_slots)
    
    # Identify consecutive time slots
    adjacency_matrix = identify_consecutive_15min_slots(fifteen_min_slots)
    
    # Create the optimization problem
    prob = LpProblem("Time_Slot_Assignment_15min", LpMaximize)
    
    # Decision variables: x[i,j] = 1 if participant i is assigned to slot j
    x = {}
    for i in range(n_participants):
        for j in range(n_slots):
            x[i, j] = LpVariable(f"x_{i}_{j}", cat='Binary')
    
    # Binary variables for slot usage: y[j] = 1 if slot j has any assignment
    y = {}
    for j in range(n_slots):
        y[j] = LpVariable(f"y_{j}", cat='Binary')
    
    # Binary variables for consecutive slot usage: z[j,k] = 1 if both slots j and k are used
    z = {}
    for j in range(n_slots):
        for k in range(n_slots):
            if adjacency_matrix[j][k] == 1:
                z[j, k] = LpVariable(f"z_{j}_{k}", cat='Binary')
    
    # Calculate max possible preference score for scaling
    max_pref_score = np.sum(np.max(preference_matrix, axis=1))
    
    # Objective: Maximize preferences + bonus for using consecutive slots
    preference_score = lpSum(preference_matrix[i, j] * x[i, j] 
                           for i in range(n_participants) 
                           for j in range(n_slots))
    
    # Contiguity bonus: reward using consecutive slots (regardless of who is assigned)
    # This encourages grouping meetings together to minimize gaps in the schedule
    contiguity_score = lpSum(z[j, k]
                           for j in range(n_slots)
                           for k in range(n_slots)
                           if adjacency_matrix[j][k] == 1)
    
    # Combine objectives
    prob += preference_score + contiguity_bonus * max_pref_score * contiguity_score
    
    # Constraints:
    # 1. Each participant gets at most one slot (exactly one 15-minute slot or none)
    for i in range(n_participants):
        prob += lpSum(x[i, j] for j in range(n_slots)) <= 1
    
    # 2. Each 15-minute slot can have at most 1 participant
    for j in range(n_slots):
        prob += lpSum(x[i, j] for i in range(n_participants)) <= 1
    
    # 3. Participants can only be assigned to available slots
    for i in range(n_participants):
        for j in range(n_slots):
            if availability_matrix[i, j] == 0:
                prob += x[i, j] == 0
    
    # 4. Link y[j] to whether slot j is used
    for j in range(n_slots):
        # If any participant is assigned to slot j, then y[j] = 1
        prob += y[j] >= lpSum(x[i, j] for i in range(n_participants)) / n_participants
        # If no participant is assigned to slot j, then y[j] = 0
        prob += y[j] <= lpSum(x[i, j] for i in range(n_participants))
    
    # 5. Link z[j,k] to consecutive slot usage
    for j in range(n_slots):
        for k in range(n_slots):
            if adjacency_matrix[j][k] == 1:
                # z[j,k] can only be 1 if both y[j] and y[k] are 1
                prob += z[j, k] <= y[j]
                prob += z[j, k] <= y[k]
                # z[j,k] should be 1 if both slots are used
                prob += z[j, k] >= y[j] + y[k] - 1
    
    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=0))  # Use msg=1 to see solver output
    
    # Extract the solution
    assignments = {}
    unassigned = []
    
    for i in range(n_participants):
        assigned = False
        for j in range(n_slots):
            if value(x[i, j]) == 1:
                assignments[participants[i]] = fifteen_min_slots[j]
                assigned = True
                break
        if not assigned:
            unassigned.append(participants[i])
    
    return assignments, unassigned

def create_assignment_report(assignments, unassigned, preference_matrix, participants, fifteen_min_slots, hour_slots):
    """
    Create a detailed report of the 15-minute assignments.
    """
    print("=== 15-MINUTE ASSIGNMENT RESULTS ===\n")
    
    # Group assignments by time slot
    slot_assignments = {}
    for participant, slot in assignments.items():
        slot_assignments[slot] = participant  # Only one per slot
    
    # Sort slots for better visualization
    assigned_slots = sorted([slot for slot in fifteen_min_slots if slot in slot_assignments], 
                          key=lambda x: fifteen_min_slots.index(x))
    
    # Report by time slot
    print("Assignments by 15-Minute Time Slot:")
    print("-" * 70)
    
    # Group by hour for cleaner display
    current_hour = None
    for slot in assigned_slots:
        # Try to identify which hour this belongs to
        hour_marker = None
        for hour_idx, hour_slot in enumerate(hour_slots):
            if hour_slot.split('[')[0].strip() in slot:
                hour_marker = hour_slot
                break
        
        if hour_marker != current_hour:
            print(f"\n{hour_marker if hour_marker else 'Time Block'}:")
            current_hour = hour_marker
        
        participant = slot_assignments[slot]
        # Get preference level
        p_idx = participants.index(participant)
        s_idx = fifteen_min_slots.index(slot)
        pref_level = preference_matrix[p_idx, s_idx]
        pref_text = "Great time!" if pref_level == 2 else "Can make it"
        
        print(f"  {slot}: {participant} ({pref_text})")
    
    # Check for contiguous blocks of assignments
    print("\n\nContiguous Assignment Blocks:")
    print("-" * 70)
    
    adjacency_matrix = identify_consecutive_15min_slots(fifteen_min_slots)
    contiguous_blocks = []
    current_block = []
    
    for i, slot in enumerate(assigned_slots):
        if not current_block:
            current_block = [slot]
        else:
            # Check if this slot is consecutive to the previous one
            prev_slot = current_block[-1]
            prev_idx = fifteen_min_slots.index(prev_slot)
            curr_idx = fifteen_min_slots.index(slot)
            
            if prev_idx < len(fifteen_min_slots) and curr_idx < len(fifteen_min_slots):
                if adjacency_matrix[prev_idx][curr_idx] == 1:
                    current_block.append(slot)
                else:
                    # Start a new block
                    if len(current_block) > 1:
                        contiguous_blocks.append(current_block)
                    current_block = [slot]
            else:
                if len(current_block) > 1:
                    contiguous_blocks.append(current_block)
                current_block = [slot]
    
    # Don't forget the last block
    if len(current_block) > 1:
        contiguous_blocks.append(current_block)
    
    if contiguous_blocks:
        print(f"Found {len(contiguous_blocks)} contiguous blocks:")
        for i, block in enumerate(contiguous_blocks):
            print(f"\n  Block {i+1}: {len(block)} consecutive 15-min slots")
            print(f"  (Each slot has a different participant)")
            for slot in block:
                participant = slot_assignments[slot]
                print(f"    {slot}: {participant}")
    else:
        print("No contiguous blocks found (all assignments are scattered)")
    
    # Report unassigned participants
    if unassigned:
        print(f"\n\nUnassigned Participants ({len(unassigned)}):")
        print("-" * 70)
        for participant in unassigned:
            print(f"  - {participant}")
    
    # Summary statistics
    print("\n\nSummary Statistics:")
    print("-" * 70)
    print(f"Total participants: {len(participants)}")
    print(f"Successfully assigned: {len(assignments)}")
    print(f"Unassigned: {len(unassigned)}")
    print(f"Total 15-minute slots available: {len(fifteen_min_slots)}")
    print(f"Slots used: {len(slot_assignments)}")
    print(f"Each participant gets: Exactly one 15-minute slot")
    
    # Count preference satisfaction
    great_time_count = 0
    can_make_count = 0
    for participant, slot in assignments.items():
        p_idx = participants.index(participant)
        s_idx = fifteen_min_slots.index(slot)
        if preference_matrix[p_idx, s_idx] == 2:
            great_time_count += 1
        else:
            can_make_count += 1
    
    print(f"\nAssignment quality:")
    print(f"  - Assigned to 'great time': {great_time_count} ({great_time_count/len(assignments)*100:.1f}%)")
    print(f"  - Assigned to 'can make it': {can_make_count} ({can_make_count/len(assignments)*100:.1f}%)")

def save_assignments_to_file(assignments, unassigned, output_path="direct_15min_assignments.csv"):
    """
    Save the 15-minute assignments to a CSV file.
    Each participant has exactly one 15-minute slot (or is unassigned).
    """
    # Create a DataFrame with the results
    results = []

    def get_slot_time(slot):
        if "[" in slot and "]" in slot:
            return slot[slot.index("[")+1:slot.index("]")]
        else:
            return slot
    
    for participant, slot in assignments.items():
        results.append({
            'Participant': participant,
            'Assigned 15-Minute Slot': get_slot_time(slot),
            'Status': 'Assigned'
        })
    
    for participant in unassigned:
        results.append({
            'Participant': participant,
            'Assigned 15-Minute Slot': 'None',
            'Status': 'Unassigned'
        })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"\n15-minute assignments saved to: {output_path}")

def print_assignments_from_csv(csv_path):
    """
    Read assignments from a CSV file and print them sorted by day and time.
    
    Args:
        csv_path: Path to the CSV file with assignments
    """
    print(f"\n=== READING ASSIGNMENTS FROM {csv_path} ===")
    print("-" * 50)
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if this is the 15-minute format or hour format
        if 'Assigned 15-Minute Slot' in df.columns:
            slot_column = 'Assigned 15-Minute Slot'
        elif 'Assigned Time Slot' in df.columns:
            slot_column = 'Assigned Time Slot'
        else:
            print("Error: Could not find assignment column in CSV")
            return
        
        # Filter only assigned participants
        assigned_df = df[df['Status'] == 'Assigned'].copy()
        
        if len(assigned_df) == 0:
            print("No assignments found in the CSV file.")
            return
        
        # Extract time slot information and parse for sorting
        slot_info = []
        for _, row in assigned_df.iterrows():
            time_slot = row[slot_column]
            participant = row['Participant']
            
            if time_slot == 'None' or pd.isna(time_slot):
                continue
            
            # Parse the time slot format
            try:
                # Handle different formats:
                # 1. "Mon 8:45p-9:00p" (15-minute format)
                # 2. "Quiz 1 Time Preference [Mon 8:30p]" (hour format)
                
                if '[' in time_slot and ']' in time_slot:
                    # Hour format: "Quiz 1 Time Preference [Mon 8:30p]"
                    bracket_content = time_slot[time_slot.index('[')+1:time_slot.index(']')]
                    parts = bracket_content.split()
                    if len(parts) >= 2:
                        day = parts[0]
                        time_str = parts[1]
                    else:
                        continue
                else:
                    # 15-minute format: "Mon 8:45p-9:00p"
                    parts = time_slot.split(' ', 1)
                    if len(parts) == 2:
                        day = parts[0]
                        time_str = parts[1].split('-')[0].strip()  # Get start time
                    else:
                        continue
                
                # Convert day to number for sorting (Mon=1, Tue=2, etc.)
                day_order = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Thurs': 4, 'Fri': 5}
                day_num = day_order.get(day, 99)  # Default to 99 for unknown days
                
                # Parse start time for sorting
                # Handle formats like "8:45p", "12:30pm"
                time_str_clean = time_str
                if time_str_clean.endswith('p') and not time_str_clean.endswith('pm'):
                    time_str_clean = time_str_clean[:-1] + 'pm'
                elif time_str_clean.endswith('a') and not time_str_clean.endswith('am'):
                    time_str_clean = time_str_clean[:-1] + 'am'
                
                # Add space before am/pm if not present
                if 'am' in time_str_clean and time_str_clean[-3] != ' ':
                    time_str_clean = time_str_clean.replace('am', ' am')
                if 'pm' in time_str_clean and time_str_clean[-3] != ' ':
                    time_str_clean = time_str_clean.replace('pm', ' pm')
                
                try:
                    start_time = datetime.strptime(time_str_clean, "%I:%M %p")
                    # Convert to minutes since midnight for sorting
                    minutes_since_midnight = start_time.hour * 60 + start_time.minute
                    
                    slot_info.append({
                        'day': day,
                        'day_num': day_num,
                        'start_time': time_str,
                        'minutes_since_midnight': minutes_since_midnight,
                        'full_slot': time_slot,
                        'participant': participant
                    })
                except:
                    # If time parsing fails, use 0 minutes
                    slot_info.append({
                        'day': day,
                        'day_num': day_num,
                        'start_time': time_str,
                        'minutes_since_midnight': 0,
                        'full_slot': time_slot,
                        'participant': participant
                    })
                    
            except Exception as e:
                # If parsing fails, add with default values
                slot_info.append({
                    'day': 'Unknown',
                    'day_num': 99,
                    'start_time': 'Unknown',
                    'minutes_since_midnight': 0,
                    'full_slot': time_slot,
                    'participant': participant
                })
        
        if len(slot_info) == 0:
            print("No valid time slots found in the assignments.")
            return
        
        # Sort by day number, then by time
        slot_info.sort(key=lambda x: (x['day_num'], x['minutes_since_midnight']))
        
        # Group by day and print
        current_day = None
        for slot in slot_info:
            if slot['day'] != current_day:
                if current_day is not None:
                    print()  # Add blank line between days
                current_day = slot['day']
                print(f"\n{current_day}:")
            
            print(f"  {slot['start_time']} - {slot['participant']}")
        
        print(f"\nTotal assigned slots: {len(slot_info)}")
        
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")

# Main execution function
def assign_15min_slots(file_path, output_path="15min_assignments.csv", contiguity_bonus=0.1):
    """
    Main function to run the 15-minute time slot assignment algorithm.
    Each participant gets exactly one 15-minute slot.
    
    Args:
        file_path: Path to the input spreadsheet with hour-long preferences
        output_path: Path for the output CSV file
        contiguity_bonus: Weight for preferring consecutive slots (0-1, default: 0.1)
                         Higher values = stronger preference for grouping participants
                         in consecutive 15-minute slots (reducing gaps between meetings)
    """
    print("Loading data...")
    df, hour_slots, participants = load_and_process_data(file_path)
    
    print(f"Found {len(participants)} participants and {len(hour_slots)} hour-long time slots")
    
    print("\nExpanding to 15-minute slots...")
    fifteen_min_slots, hour_to_15min_map = expand_to_15min_slots(hour_slots)
    print(f"Created {len(fifteen_min_slots)} fifteen-minute slots")
    
    print("\nProcessing preferences...")
    preference_matrix, availability_matrix = create_expanded_preference_matrix(
        df, hour_slots, participants, hour_to_15min_map, len(fifteen_min_slots)
    )
    
    print("\nSolving assignment problem...")
    assignments, unassigned = solve_assignment(
        participants, fifteen_min_slots, preference_matrix, availability_matrix, contiguity_bonus
    )
    
    print("\nGenerating report...")
    create_assignment_report(assignments, unassigned, preference_matrix, participants, fifteen_min_slots, hour_slots)
    
    save_assignments_to_file(assignments, unassigned, output_path)
    
    return assignments, unassigned

# Example usage
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Assign participants to time slots based on preferences')
    parser.add_argument('--file_path', default='participant_availability.xlsx', help='Path to the Excel/CSV file with participant availability data')
    parser.add_argument('--max-per-slot', type=int, default=4, 
                       help='Maximum number of participants per time slot (default: 4)')
    parser.add_argument('--output', default='time_slot_assignments_new.csv',
                       help='Output CSV file path (default: time_slot_assignments.csv)')
    parser.add_argument('--contiguity-bonus', type=float, default=0.05, 
                       help='Bonus weight for preferring consecutive time slots (default: 0.05)')
    parser.add_argument('--read-assignments', 
                       help='Read and display assignments from a CSV file (instead of running assignment algorithm)')
    
    args = parser.parse_args()
    
    if args.read_assignments:
        # Read and display existing assignments
        print_assignments_from_csv(args.read_assignments)
    else:
        # Run the assignment algorithm
        assignments, unassigned = assign_15min_slots(
            file_path=args.file_path,
            output_path=args.output,
            contiguity_bonus=args.contiguity_bonus
        )