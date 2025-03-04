import json
import os
import argparse

def process_file(input_file, output_file):
    """
    Reads a text file, extracts structured data, and saves it in JSONL format.
    
    Args:
        input_file (str): Path to the input text file.
        output_file (str): Path to the output JSONL file.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    preprocessed_data = []
    line_idx = 0

    for line in lines:
        line = line.strip()
        components = line.split('||')

        try:
            src = components[0].strip()
            rationales_trg = components[1].strip().split('####')

            if len(rationales_trg) == 2:
                rationales, trg = rationales_trg
                preprocessed_line = {
                    'src': src,
                    'rationales': rationales.strip(),
                    'trg': trg.strip()
                }
                preprocessed_data.append(preprocessed_line)
            else:
                print(f"Skipping malformed line {line_idx}: {line}")

        except Exception as e:
            print(f"Error processing line {line_idx}: {line} | Error: {e}")

        finally:
            line_idx += 1

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save processed data in JSONL format
    with open(output_file, 'w', encoding='utf-8') as file:
        for line in preprocessed_data:
            json.dump(line, file)
            file.write('\n')

    print(f"Processed {len(preprocessed_data)} valid lines from {input_file} -> Saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process text files and convert them into JSONL format.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the directory containing input text files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save the processed JSONL files.")

    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Process all three files
    for file_name in ['train.txt', 'test.txt', 'valid.txt']:
        input_file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name.replace('.txt', '.jsonl'))
        process_file(input_file_path, output_file_path)
