import time
import os
import sys

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

output_file = r"C:\Users\barla\AppData\Local\Temp\claude\c--Users-barla-Downloads\tasks\b3ca160.output"
last_size = 0

print("Monitoring MCH Experiment Progress...")
print("=" * 60)
print("Press Ctrl+C to stop monitoring (experiment continues in background)")
print("=" * 60 + "\n")

try:
    while True:
        if os.path.exists(output_file):
            current_size = os.path.getsize(output_file)
            if current_size > last_size:
                with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_size)
                    new_content = f.read()
                    if new_content:
                        print(new_content, end='')
                        last_size = current_size
        time.sleep(5)
except KeyboardInterrupt:
    print("\n\nMonitoring stopped. Experiment continues in background.")
    print("Results will be saved to: mch_results_sequential_30trials.json")
