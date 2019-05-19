import subprocess

queue_command = "squeue | grep brianlui"
output = subprocess.check_output(queue_command.split())
print(output)

lines = output.strip().splitlines()
for line in lines:
    job_id = line.split()[0]
    tail_command = "tail -1 save-" + job_id + ".out"
    output = subprocess.check_output(tail_command.split(), stdout=subprocess.PIPE)
    print(output)