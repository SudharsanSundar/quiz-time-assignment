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
    print('TIME SLOTS: ', len(time_slots), time_slots)
    
    # Get participant identifiers (assuming first column is name/ID)
    participants = df.iloc[:, 1].tolist()
    print('PARTICIPANTS: ', len(participants), participants)
    
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

def solve_assignment(participants, time_slots, preference_matrix, availability_matrix, max_per_slot=4):
    """
    Solve the assignment problem using linear programming.
    
    Returns:
        assignments: Dictionary mapping participants to their assigned time slots
        unassigned: List of participants who couldn't be assigned
    """
    n_participants = len(participants)
    n_slots = len(time_slots)
    
    # Create the optimization problem
    prob = LpProblem("Time_Slot_Assignment", LpMaximize)
    
    # Decision variables: x[i,j] = 1 if participant i is assigned to slot j
    x = {}
    for i in range(n_participants):
        for j in range(n_slots):
            x[i, j] = LpVariable(f"x_{i}_{j}", cat='Binary')
    
    # Objective: Maximize the sum of preferences for assigned slots
    prob += lpSum(preference_matrix[i, j] * x[i, j] 
                  for i in range(n_participants) 
                  for j in range(n_slots))
    
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
    
    # Report by time slot
    print("Assignments by Time Slot:")
    print("-" * 50)
    for slot in time_slots:
        if slot in slot_assignments:
            print(f"\n{slot}: ({len(slot_assignments[slot])} participants)")
            for participant in slot_assignments[slot]:
                # Get preference level for this assignment
                p_idx = participants.index(participant)
                s_idx = time_slots.index(slot)
                pref_level = preference_matrix[p_idx, s_idx]
                pref_text = "Great time!" if pref_level == 2 else "Can make it"
                print(f"  - {participant} ({pref_text})")
    
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
def assign_time_slots(file_path, max_per_slot=4, output_path="assignments.csv"):
    """
    Main function to run the time slot assignment algorithm.
    
    Args:
        file_path: Path to the input spreadsheet
        max_per_slot: Maximum number of participants per time slot (default: 4)
        output_path: Path for the output CSV file
    """
    print("Loading data...")
    df, time_slots, participants = load_and_process_data(file_path)
    
    print(f"Found {len(participants)} participants and {len(time_slots)} time slots")
    
    print("\nProcessing preferences...")
    preference_matrix, availability_matrix = create_preference_matrix(df, time_slots, participants)
    
    print("\nSolving assignment problem...")
    assignments, unassigned = solve_assignment(
        participants, time_slots, preference_matrix, availability_matrix, max_per_slot
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
    parser.add_argument('--output', default='time_slot_assignments.csv',
                       help='Output CSV file path (default: time_slot_assignments.csv)')
    
    args = parser.parse_args()
    
    # Run the assignment
    assignments, unassigned = assign_time_slots(
        file_path=args.file_path,
        max_per_slot=args.max_per_slot,
        output_path=args.output
    )