import pandas as pd
import random
import re
from datetime import datetime, timedelta
import argparse

def parse_time_slot(time_slot_str):
    """
    Parse a time slot string to extract start time and day.
    Handles format like "Quiz 1 Time Preference [Mon 8:30p]"
    
    Returns:
        start_time: datetime object (using a dummy date)
        day: string (e.g., "Mon")
        original_format: dict with parsing info for formatting output
    """
    import re
    
    # Extract the bracketed content
    match = re.search(r'\[(.*?)\]', time_slot_str)
    if not match:
        raise ValueError(f"Could not find bracketed time in: {time_slot_str}")
    
    bracket_content = match.group(1).strip()
    parts = bracket_content.split()
    
    if len(parts) != 2:
        raise ValueError(f"Unexpected format in brackets: {bracket_content}")
    
    day = parts[0]
    time_str = parts[1]
    
    # Parse the time - handle formats like "8:30p", "12:30pm", "4:30p"
    # Normalize the time format
    if time_str.endswith('p') and not time_str.endswith('pm'):
        time_str = time_str[:-1] + 'pm'
    elif time_str.endswith('a') and not time_str.endswith('am'):
        time_str = time_str[:-1] + 'am'
    
    # Add space before am/pm if not present
    if 'am' in time_str and time_str[-3] != ' ':
        time_str = time_str.replace('am', ' am')
    if 'pm' in time_str and time_str[-3] != ' ':
        time_str = time_str.replace('pm', ' pm')
    
    # Parse the time
    try:
        start_time = datetime.strptime(time_str, "%I:%M %p")
    except:
        try:
            start_time = datetime.strptime(time_str, "%H:%M %p")
        except:
            raise ValueError(f"Could not parse time: {time_str}")
    
    # Store original format info
    original_format = {
        'prefix': time_slot_str.split('[')[0].strip(),
        'day_format': day,
        'time_format': parts[1]  # Keep original format
    }
    
    return start_time, day, original_format

def format_15min_slot(base_time, minutes_offset, day, original_format):
    """
    Format a 15-minute slot string in the same style as the input.
    
    Args:
        base_time: datetime object for the start of the hour slot
        minutes_offset: 0, 15, 30, or 45
        day: day of the week string
        original_format: dict with formatting preferences
    
    Returns:
        Formatted string like "Quiz 1 Time Preference [Mon 8:45p-9:00p]"
    """
    start = base_time + timedelta(minutes=minutes_offset)
    end = start + timedelta(minutes=15)
    
    # Format times to match the original style
    # Check if original used 'p' or 'pm'
    use_short_suffix = original_format['time_format'].endswith('p') and not original_format['time_format'].endswith('pm')
    
    if use_short_suffix:
        # Use short format (8:30p)
        start_str = start.strftime("%I:%M%p").lstrip("0").lower()
        end_str = end.strftime("%I:%M%p").lstrip("0").lower()
        # Convert 'pm' to 'p' and 'am' to 'a'
        start_str = start_str.replace('pm', 'p').replace('am', 'a')
        end_str = end_str.replace('pm', 'p').replace('am', 'a')
    else:
        # Use full format (8:30pm)
        start_str = start.strftime("%I:%M%p").lstrip("0").lower()
        end_str = end.strftime("%I:%M%p").lstrip("0").lower()
    
    # Format: "Quiz 1 Time Preference [Mon 8:45p-9:00p]"
    return f"{day} {start_str}-{end_str}"

def assign_15min_blocks(input_csv="time_slot_assignments.csv", 
                       output_csv="15min_assignments.csv",
                       randomize_order=True):
    """
    Assign respondents to 15-minute blocks within their hour-long time slots.
    
    Expects input CSV with "Assigned Time Slot" in format:
    - "Quiz 1 Time Preference [Mon 8:30p]"
    - "Quiz 1 Time Preference [Fri 12:30pm]"
    - "Quiz 1 Time Preference [Thurs 5:30p]"
    
    Args:
        input_csv: Path to the CSV file from the time slot assignment
        output_csv: Path for the output CSV with 15-minute assignments
        randomize_order: If True, randomize assignment order within each slot
    
    Returns:
        DataFrame with the 15-minute assignments
    """
    # Read the assignments
    df = pd.read_csv(input_csv)
    
    # Filter only assigned participants
    assigned_df = df[df['Status'] == 'Assigned'].copy()
    
    # Group by time slot
    grouped = assigned_df.groupby('Assigned Time Slot')
    
    # Define the assignment patterns
    assignment_patterns = {
        1: [30],           # 1 person: middle of hour
        2: [15, 30],       # 2 people: +0:15 and +0:30
        3: [0, 15, 30],    # 3 people: +0:00, +0:15, +0:30
        4: [0, 15, 30, 45] # 4 people: all slots
    }
    
    # Store results
    results = []
    
    # Process each time slot group
    for time_slot, group in grouped:
        participants = group['Participant'].tolist()
        n_participants = len(participants)
        
        # Get the appropriate pattern
        pattern = assignment_patterns[n_participants]
        
        # Randomize participant order if requested
        if randomize_order:
            participants = participants.copy()
            random.shuffle(participants)
        
        # Parse the time slot
        try:
            base_time, day, original_format = parse_time_slot(time_slot)
            
            # Assign each participant to their 15-minute block
            for i, participant in enumerate(participants):
                minutes_offset = pattern[i]
                slot_15min = format_15min_slot(base_time, minutes_offset, day, original_format)
                
                results.append({
                    'Participant': participant,
                    'Original Hour Slot': time_slot,
                    '15-Minute Slot': slot_15min,
                    'Start Offset': f"+{minutes_offset:02d}",
                    'Status': 'Assigned'
                })
                
        except Exception as e:
            print(f"Warning: Could not parse time slot '{time_slot}': {e}")
            # Fallback: just append the offset to the original slot
            for i, participant in enumerate(participants):
                minutes_offset = pattern[i]
                results.append({
                    'Participant': participant,
                    'Original Hour Slot': time_slot,
                    '15-Minute Slot': f"{time_slot} +{minutes_offset:02d}min",
                    'Start Offset': f"+{minutes_offset:02d}",
                    'Status': 'Assigned'
                })
    
    # Add unassigned participants
    unassigned_df = df[df['Status'] == 'Unassigned']
    for _, row in unassigned_df.iterrows():
        results.append({
            'Participant': row['Participant'],
            'Original Hour Slot': 'None',
            '15-Minute Slot': 'None',
            'Start Offset': 'N/A',
            'Status': 'Unassigned'
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    save_df = results_df[['Participant', '15-Minute Slot']]
    save_df.to_csv(output_csv, index=False)
    
    # Print summary
    print_assignment_summary(results_df)
    
    # Print all assigned slots in chronological order
    print_all_assigned_slots(results_df)
    
    return results_df

def print_assignment_summary(results_df):
    """
    Print a summary of the 15-minute assignments.
    """
    print("=== 15-MINUTE ASSIGNMENT SUMMARY ===\n")
    
    # Group by original hour slot
    assigned_df = results_df[results_df['Status'] == 'Assigned']
    
    for hour_slot in assigned_df['Original Hour Slot'].unique():
        slot_data = assigned_df[assigned_df['Original Hour Slot'] == hour_slot]
        print(f"\n{hour_slot}:")
        print("-" * 70)
        
        # Sort by start offset
        slot_data = slot_data.sort_values('Start Offset')
        
        for _, row in slot_data.iterrows():
            # Extract just the time range from the 15-minute slot
            match = re.search(r'\[(.*?)\]', row['15-Minute Slot'])
            if match:
                time_info = match.group(1)
                print(f"  {row['Start Offset']} - {row['Participant']} ({time_info})")
            else:
                print(f"  {row['Start Offset']} - {row['Participant']}")
    
    # Summary statistics
    print("\n\nSummary Statistics:")
    print("-" * 50)
    total_assigned = len(results_df[results_df['Status'] == 'Assigned'])
    total_unassigned = len(results_df[results_df['Status'] == 'Unassigned'])
    
    print(f"Total participants assigned to 15-min slots: {total_assigned}")
    print(f"Unassigned participants: {total_unassigned}")
    
    # Count distribution
    offset_counts = assigned_df['Start Offset'].value_counts().sort_index()
    print("\nDistribution by start time offset:")
    for offset, count in offset_counts.items():
        print(f"  {offset} minutes: {count} participants")

def create_calendar_format(results_df, output_csv="calendar_import.csv"):
    """
    Create a CSV file formatted for calendar import.
    """
    import re
    
    calendar_data = []
    
    assigned_df = results_df[results_df['Status'] == 'Assigned']
    
    for _, row in assigned_df.iterrows():
        # Parse the 15-minute slot format: "Quiz 1 Time Preference [Mon 8:45p-9:00p]"
        match = re.search(r'\[(.*?)\]', row['15-Minute Slot'])
        if match:
            bracket_content = match.group(1)
            parts = bracket_content.split(' ', 1)
            
            if len(parts) == 2:
                day = parts[0]
                time_range = parts[1]
                
                # Split the time range
                if '-' in time_range:
                    start_time, end_time = time_range.split('-')
                    
                    calendar_data.append({
                        'Subject': f"Meeting with {row['Participant']}",
                        'Start Time': start_time.strip(),
                        'End Time': end_time.strip(),
                        'Day': day,
                        'Description': f"15-minute meeting with {row['Participant']}",
                        'Original Slot': row['Original Hour Slot']
                    })
    
    calendar_df = pd.DataFrame(calendar_data)
    calendar_df.to_csv(output_csv, index=False)
    print(f"\nCalendar import file saved to: {output_csv}")
    
    return calendar_df

def print_all_assigned_slots(results_df):
    """
    Print all assigned 15-minute time slots in chronological order.
    """
    print("\n=== ALL ASSIGNED 15-MINUTE TIME SLOTS ===")
    print("(Sorted by day and time)")
    print("-" * 50)
    
    # Filter only assigned participants
    assigned_df = results_df[results_df['Status'] == 'Assigned'].copy()
    
    if len(assigned_df) == 0:
        print("No assignments found.")
        return
    
    # Extract time slot information and parse for sorting
    slot_info = []
    for _, row in assigned_df.iterrows():
        time_slot = row['15-Minute Slot']
        
        # Parse the time slot format: "Mon 8:45p-9:00p"
        try:
            # Split by space to get day and time range
            parts = time_slot.split(' ', 1)
            if len(parts) == 2:
                day = parts[0]
                time_range = parts[1]
                
                # Split time range to get start time
                if '-' in time_range:
                    start_time_str = time_range.split('-')[0].strip()
                    
                    # Convert day to number for sorting (Mon=1, Tue=2, etc.)
                    day_order = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Thurs': 4, 'Fri': 5}
                    day_num = day_order.get(day, 99)  # Default to 99 for unknown days
                    
                    # Parse start time for sorting
                    # Handle formats like "8:45p", "12:30pm"
                    time_str = start_time_str
                    if time_str.endswith('p') and not time_str.endswith('pm'):
                        time_str = time_str[:-1] + 'pm'
                    elif time_str.endswith('a') and not time_str.endswith('am'):
                        time_str = time_str[:-1] + 'am'
                    
                    # Add space before am/pm if not present
                    if 'am' in time_str and time_str[-3] != ' ':
                        time_str = time_str.replace('am', ' am')
                    if 'pm' in time_str and time_str[-3] != ' ':
                        time_str = time_str.replace('pm', ' pm')
                    
                    try:
                        start_time = datetime.strptime(time_str, "%I:%M %p")
                        # Convert to minutes since midnight for sorting
                        minutes_since_midnight = start_time.hour * 60 + start_time.minute
                        
                        slot_info.append({
                            'day': day,
                            'day_num': day_num,
                            'start_time': start_time_str,
                            'minutes_since_midnight': minutes_since_midnight,
                            'full_slot': time_slot,
                            'participant': row['Participant']
                        })
                    except:
                        # If time parsing fails, use 0 minutes
                        slot_info.append({
                            'day': day,
                            'day_num': day_num,
                            'start_time': start_time_str,
                            'minutes_since_midnight': 0,
                            'full_slot': time_slot,
                            'participant': row['Participant']
                        })
        except:
            # If parsing fails, add with default values
            slot_info.append({
                'day': 'Unknown',
                'day_num': 99,
                'start_time': 'Unknown',
                'minutes_since_midnight': 0,
                'full_slot': time_slot,
                'participant': row['Participant']
            })
    
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

# Example usage and testing
# TODO: Some times are not parsed right
# TODO: Make save value simpler and just the info we want --> Something we can post to Ed
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Assign participants to 15-minute blocks within their hour-long time slots')
    parser.add_argument('--input', default='time_slot_assignments.csv',
                       help='Path to the input CSV file from time slot assignment (default: time_slot_assignments.csv)')
    parser.add_argument('--output', default='15min_assignments.csv',
                       help='Path for the output CSV with 15-minute assignments (default: 15min_assignments.csv)')
    parser.add_argument('--calendar-output', default='calendar_import.csv',
                       help='Path for the calendar import CSV file (default: calendar_import.csv)')
    parser.add_argument('--no-randomize', action='store_true',
                       help='Disable randomization of assignment order within each slot')
    
    args = parser.parse_args()
    
    # Process the assignments
    results = assign_15min_blocks(
        input_csv=args.input,
        output_csv=args.output,
        randomize_order=not args.no_randomize
    )
    
    # Optional: Create calendar import format
    calendar_df = create_calendar_format(
        results, 
        output_csv=args.calendar_output
    )
    
    # Example: View assignments for a specific time slot
    print("\n\nExample - Viewing specific time slot assignments:")
    print("-" * 50)
    
    # Get unique hour slots
    hour_slots = results[results['Status'] == 'Assigned']['Original Hour Slot'].unique()
    if len(hour_slots) > 0:
        example_slot = hour_slots[0]
        slot_assignments = results[results['Original Hour Slot'] == example_slot]
        
        print(f"\nAssignments for {example_slot}:")
        for _, row in slot_assignments.iterrows():
            print(f"  {row['Participant']}: {row['15-Minute Slot']}")