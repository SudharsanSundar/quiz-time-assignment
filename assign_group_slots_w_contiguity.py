import pandas as pd
import numpy as np
from pulp import *
import openpyxl
import argparse

def load_and_process_data(file_path):
    """
    Load the spreadsheet and process preference data.
    
    Args:
        file_path: Path to the Excel/CSV file
    
    Returns:
        df: DataFrame with participant data
        time_slots: List of time slot column names (C through BF)
        participants: List of participant names/IDs
    """
    # Load the data
    df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
    
    # Get time slot columns (C through BF)
    # Excel columns C through BF correspond to indices 2 through 57
    time_slot_cols = df.columns[2:58]  # Adjust if your column range is different
    time_slots = list(time_slot_cols)
    
    # Get participant identifiers (assuming first column is name/ID)
    participants = df.iloc[:, 0].tolist()
    
    return df, time_slots, participants

def create_preference_matrix(df, time_slots, participants):
    """
    Create a preference matrix where:
    - 2 = "This is a great time for me!"
    - 1 = "I can make it at this time."
    - 0 = "I cannot make this time."
    
    Returns:
        preference_matrix: 2D numpy array of preferences
        availability_matrix: Binary matrix (1 if available, 0 if not)
    """
    n_participants = len(participants)
    n_slots = len(time_slots)
    
    preference_matrix = np.zeros((n_participants, n_slots))
    availability_matrix = np.zeros((n_participants, n_slots))
    
    for i, participant in enumerate(participants):
        for j, slot in enumerate(time_slots):
            pref = df.iloc[i][slot]
            
            if pref == "This is a great time for me!":
                preference_matrix[i, j] = 2
                availability_matrix[i, j] = 1
            elif pref == "I can make it at this time.":
                preference_matrix[i, j] = 1
                availability_matrix[i, j] = 1
            else:  # "I cannot make this time."
                preference_matrix[i, j] = 0
                availability_matrix[i, j] = 0
    
    return preference_matrix, availability_matrix

def identify_consecutive_slots(time_slots):
    """
    Identify which time slots are consecutive (adjacent in time).
    
    Returns:
        adjacency_matrix: n_slots x n_slots matrix where adjacency_matrix[i][j] = 1 
                         if slot j immediately follows slot i
    """
    n_slots = len(time_slots)
    adjacency_matrix = np.zeros((n_slots, n_slots))
    
    # Parse time slots to identify consecutive ones
    slot_info = []
    for slot in time_slots:
        # Extract day and time from slot name
        # Assuming format contains day abbreviations and times
        day_order = {'Mon': 0, 'Monday': 0, 'Tue': 1, 'Tues': 1, 'Tuesday': 1, 
                     'Wed': 2, 'Wednesday': 2, 'Thu': 3, 'Thur': 3, 'Thurs': 3, 'Thursday': 3,
                     'Fri': 4, 'Friday': 4, 'Sat': 5, 'Saturday': 5, 'Sun': 6, 'Sunday': 6}
        
        # Try to extract day and hour from the slot string
        slot_lower = slot.lower()
        day_num = None
        hour = None
        
        # Find day
        for day_name, num in day_order.items():
            if day_name.lower() in slot_lower:
                day_num = num
                break
        
        # Find time (look for patterns like "8:30", "12:00", etc.)
        import re
        time_pattern = r'(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.|)'
        time_match = re.search(time_pattern, slot_lower)
        if time_match:
            hour_str = time_match.group(1)
            minute_str = time_match.group(2)
            period = time_match.group(3)
            
            hour = int(hour_str)
            minute = int(minute_str)
            
            # Convert to 24-hour format
            if period and ('pm' in period or 'p.m.' in period) and hour != 12:
                hour += 12
            elif period and ('am' in period or 'a.m.' in period) and hour == 12:
                hour = 0
            
            # Convert to minutes since start of week
            if day_num is not None:
                total_minutes = day_num * 24 * 60 + hour * 60 + minute
            else:
                total_minutes = hour * 60 + minute
        else:
            total_minutes = None
            
        slot_info.append((slot, day_num, total_minutes))
    
    # Build adjacency matrix
    for i in range(n_slots):
        for j in range(n_slots):
            if i != j and slot_info[i][2] is not None and slot_info[j][2] is not None:
                # Check if slot j is exactly one hour after slot i
                if slot_info[j][2] - slot_info[i][2] == 60:
                    adjacency_matrix[i][j] = 1
    
    return adjacency_matrix

def solve_assignment(participants, time_slots, preference_matrix, availability_matrix, max_per_slot=4, contiguity_bonus=0.1):
    """
    Solve the assignment problem using linear programming with preference for contiguous assignments.
    
    Args:
        participants: List of participant names
        time_slots: List of time slot names
        preference_matrix: Matrix of preferences
        availability_matrix: Binary matrix of availability
        max_per_slot: Maximum participants per slot
        contiguity_bonus: Bonus weight for using consecutive time slots (0-1, default 0.1)
    
    Returns:
        assignments: Dictionary mapping participants to their assigned time slots
        unassigned: List of participants who couldn't be assigned
    """
    n_participants = len(participants)
    n_slots = len(time_slots)
    
    # Identify consecutive time slots
    adjacency_matrix = identify_consecutive_slots(time_slots)
    
    # Create the optimization problem
    prob = LpProblem("Time_Slot_Assignment", LpMaximize)
    
    # Decision variables: x[i,j] = 1 if participant i is assigned to slot j
    x = {}
    for i in range(n_participants):
        for j in range(n_slots):
            x[i, j] = LpVariable(f"x_{i}_{j}", cat='Binary')
    
    # Binary variables for slot usage: y[j] = 1 if slot j has any assignments
    y = {}
    for j in range(n_slots):
        y[j] = LpVariable(f"y_{j}", cat='Binary')
    
    # Binary variables for consecutive slot pairs: z[j,k] = 1 if both slots j and k are used and k follows j
    z = {}
    for j in range(n_slots):
        for k in range(n_slots):
            if adjacency_matrix[j][k] == 1:
                z[j, k] = LpVariable(f"z_{j}_{k}", cat='Binary')
    
    # Calculate max possible preference score for scaling
    max_pref_score = np.sum(np.max(preference_matrix, axis=1))
    
    # Objective: Maximize preferences + bonus for contiguous assignments
    preference_score = lpSum(preference_matrix[i, j] * x[i, j] 
                           for i in range(n_participants) 
                           for j in range(n_slots))
    
    # Contiguity bonus: reward using consecutive slots
    contiguity_score = lpSum(z[j, k] 
                           for j in range(n_slots) 
                           for k in range(n_slots) 
                           if adjacency_matrix[j][k] == 1)
    
    # Combine objectives (scale contiguity bonus relative to preferences)
    prob += preference_score + contiguity_bonus * max_pref_score * contiguity_score
    
    # Constraints:
    # 1. Each participant can only be assigned to one slot
    for i in range(n_participants):
        prob += lpSum(x[i, j] for j in range(n_slots)) <= 1
    
    # 2. Each slot can have at most max_per_slot participants
    for j in range(n_slots):
        prob += lpSum(x[i, j] for i in range(n_participants)) <= max_per_slot
    
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
                # z[j,k] should be 1 if both slots are used (soft constraint via objective)
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
                assignments[participants[i]] = time_slots[j]
                assigned = True
                break
        if not assigned:
            unassigned.append(participants[i])
    
    return assignments, unassigned

def create_assignment_report(assignments, unassigned, preference_matrix, participants, time_slots):
    """
    Create a detailed report of the assignments.
    """
    print("=== ASSIGNMENT RESULTS ===\n")
    
    # Group assignments by time slot
    slot_assignments = {}
    for participant, slot in assignments.items():
        if slot not in slot_assignments:
            slot_assignments[slot] = []
        slot_assignments[slot].append(participant)
    
    # Sort slots by time for better visualization of contiguous assignments
    sorted_slots = sorted(slot_assignments.keys())
    
    # Report by time slot
    print("Assignments by Time Slot:")
    print("-" * 50)
    
    # Check for contiguous assignments
    adjacency_matrix = identify_consecutive_slots(time_slots)
    contiguous_blocks = []
    current_block = []
    
    for i, slot in enumerate(sorted_slots):
        if slot in slot_assignments:
            if current_block and i > 0:
                # Check if this slot is consecutive to the previous one
                prev_slot = sorted_slots[i-1]
                prev_idx = time_slots.index(prev_slot) if prev_slot in time_slots else -1
                curr_idx = time_slots.index(slot) if slot in time_slots else -1
                
                if prev_idx >= 0 and curr_idx >= 0 and adjacency_matrix[prev_idx][curr_idx] == 1:
                    current_block.append(slot)
                else:
                    # Start a new block
                    if len(current_block) > 1:
                        contiguous_blocks.append(current_block)
                    current_block = [slot]
            else:
                current_block = [slot]
                
            print(f"\n{slot}: ({len(slot_assignments[slot])} participants)")
            for participant in slot_assignments[slot]:
                # Get preference level for this assignment
                p_idx = participants.index(participant)
                s_idx = time_slots.index(slot)
                pref_level = preference_matrix[p_idx, s_idx]
                pref_text = "Great time!" if pref_level == 2 else "Can make it"
                print(f"  - {participant} ({pref_text})")
    
    # Add last block if it exists
    if len(current_block) > 1:
        contiguous_blocks.append(current_block)
    
    # Report unassigned participants
    if unassigned:
        print(f"\n\nUnassigned Participants ({len(unassigned)}):")
        print("-" * 50)
        for participant in unassigned:
            print(f"  - {participant}")
            # Show their available slots
            p_idx = participants.index(participant)
            available_slots = [time_slots[j] for j in range(len(time_slots)) 
                             if preference_matrix[p_idx, j] > 0]
            if available_slots:
                print(f"    Available slots: {', '.join(available_slots[:5])}")
                if len(available_slots) > 5:
                    print(f"    ... and {len(available_slots) - 5} more")
    
    # Summary statistics
    print("\n\nSummary Statistics:")
    print("-" * 50)
    print(f"Total participants: {len(participants)}")
    print(f"Successfully assigned: {len(assignments)}")
    print(f"Unassigned: {len(unassigned)}")
    
    # Count preference satisfaction
    great_time_count = 0
    can_make_count = 0
    for participant, slot in assignments.items():
        p_idx = participants.index(participant)
        s_idx = time_slots.index(slot)
        if preference_matrix[p_idx, s_idx] == 2:
            great_time_count += 1
        else:
            can_make_count += 1
    
    print(f"\nAssignment quality:")
    print(f"  - Assigned to 'great time': {great_time_count} ({great_time_count/len(assignments)*100:.1f}%)")
    print(f"  - Assigned to 'can make it': {can_make_count} ({can_make_count/len(assignments)*100:.1f}%)")
    
    # Report contiguous assignments
    if contiguous_blocks:
        print(f"\nContiguous assignment blocks found: {len(contiguous_blocks)}")
        for i, block in enumerate(contiguous_blocks):
            total_in_block = sum(len(slot_assignments[slot]) for slot in block)
            print(f"  Block {i+1}: {len(block)} consecutive slots with {total_in_block} total participants")
            for slot in block:
                print(f"    - {slot} ({len(slot_assignments[slot])} participants)")

def save_assignments_to_file(assignments, unassigned, output_path="assignments.csv"):
    """
    Save the assignments to a CSV file.
    """
    # Create a DataFrame with the results
    results = []
    
    for participant, slot in assignments.items():
        results.append({
            'Participant': participant,
            'Assigned Time Slot': slot,
            'Status': 'Assigned'
        })
    
    for participant in unassigned:
        results.append({
            'Participant': participant,
            'Assigned Time Slot': 'None',
            'Status': 'Unassigned'
        })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"\nAssignments saved to: {output_path}")

# Main execution function
def assign_time_slots(file_path, max_per_slot=4, output_path="assignments.csv", contiguity_bonus=0.1):
    """
    Main function to run the time slot assignment algorithm with preference for contiguous assignments.
    
    Args:
        file_path: Path to the input spreadsheet
        max_per_slot: Maximum number of participants per time slot (default: 4)
        output_path: Path for the output CSV file
        contiguity_bonus: Weight for preferring consecutive time slots (0-1, default: 0.1)
                         Higher values = stronger preference for grouping assignments
                         0 = no preference for contiguous assignments
                         0.1 = mild preference (recommended)
                         0.3 = strong preference
    """
    print("Loading data...")
    df, time_slots, participants = load_and_process_data(file_path)
    
    print(f"Found {len(participants)} participants and {len(time_slots)} time slots")
    
    print("\nProcessing preferences...")
    preference_matrix, availability_matrix = create_preference_matrix(df, time_slots, participants)
    
    print("\nSolving assignment problem with contiguity preference...")
    assignments, unassigned = solve_assignment(
        participants, time_slots, preference_matrix, availability_matrix, 
        max_per_slot, contiguity_bonus
    )
    
    print("\nGenerating report...")
    create_assignment_report(assignments, unassigned, preference_matrix, participants, time_slots)
    
    save_assignments_to_file(assignments, unassigned, output_path)
    
    return assignments, unassigned

# Example usage
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Assign participants to time slots based on preferences')
    parser.add_argument('file_path', help='Path to the Excel/CSV file with participant availability data')
    parser.add_argument('--max-per-slot', type=int, default=4, 
                       help='Maximum number of participants per time slot (default: 4)')
    parser.add_argument('--output', default='time_slot_assignments_new.csv',
                       help='Output CSV file path (default: time_slot_assignments.csv)')
    parser.add_argument('--contiguity-bonus', type=float, default=0.1, 
                       help='Bonus weight for preferring consecutive time slots (default: 0.1)')
    
    args = parser.parse_args()
    
    # Run the assignment
    assignments, unassigned = assign_time_slots(
        file_path=args.file_path,
        max_per_slot=args.max_per_slot,
        output_path=args.output,
        contiguity_bonus=args.contiguity_bonus
    )