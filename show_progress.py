import subprocess

queue_command = "squeue"
queue_output = subprocess.check_output(['squeue'])
grep_output = subprocess.check_output(['grep', 'brianlui'], stdin=queue_output)
print(grep_output)

lines = grep_output.strip().splitlines()
for line in lines:
    job_id = line.split()[0]
    tail_command = "tail -1 save-" + job_id + ".out"
    output = subprocess.check_output(tail_command.split())
    print(output)