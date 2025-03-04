import jsonlines
import argparse

def extract_patterns(string):
    ans = string.split('<|endoftext|>')[0]
    ans = ans.split('####')[-1]
    ans = ans.strip()
    if ans.isnumeric():
        return ans
    else:
        return None

def calculate_correct_rate(jsonl_file):
    total_lines = 0
    correct_lines = 0

    try:
        with open(jsonl_file, 'r') as file:
            for data in jsonlines.Reader(file):
                recover = data.get('recover', '')
                source = data.get('source', '')

                if '<|endoftext|>||' in recover:
                    recover = recover.split('<|endoftext|>||')[1]
                    if '<|endoftext|>' in recover:
                        recover = recover.strip()
                        ans = extract_patterns(recover)

                        if ans and len(ans) != 0:
                            try:
                                reference = source.split('<|endoftext|>||')[1]
                                reference = reference.split('<|endoftext|>')[0]
                                reference = extract_patterns(reference.strip())
                                if reference and reference.isnumeric():
                                    if ans == reference:
                                        correct_lines += 1
                                else:
                                    print(f"Bad reference detected: reference: {source}\nrecover: {recover}")
                            except Exception as e:
                                print(f"Bad reference detected: reference: {source}\nrecover: {recover}\nError: {e}")
                                continue

                total_lines += 1

        correct_rate = (correct_lines / total_lines) * 100 if total_lines > 0 else 0
        print(f'Correct lines: {correct_lines}\nTotal lines: {total_lines}')
        return correct_rate

    except FileNotFoundError:
        print(f"Error: File {jsonl_file} not found.")
        return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the SEDD generated answers.")
    parser.add_argument('--generated_output_path', type=str, required=True, help="Base path to generated answers.")
    parser.add_argument('--steps', type=int, nargs='+', required=True, help="List of steps of generated answers (e.g., 4 8 16 32 64).")
    
    args = parser.parse_args()
    generated_output_path = args.generated_output_path
    steps = args.steps

    acc = []
    for step in steps:
        result_path = f'{generated_output_path}/step_{step}.jsonl'
        correct_rate = calculate_correct_rate(result_path)
        acc.append(correct_rate)

    for i in range(len(steps)):
        print(f"Step {steps[i]} accuracy: {acc[i]}")

