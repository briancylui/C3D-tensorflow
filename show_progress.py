from subprocess import PIPE, STDOUT, Popen, check_output

pipe = Popen(['grep', 'brianlui'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
queue_output = check_output(['squeue'])
grep_output = pipe.communicate(input=queue_output)[0]
print(grep_output)

lines = grep_output.strip().splitlines()
for line in lines:
    job_id = line.split()[0]
    tail_command = "tail -1 save-" + job_id + ".out"
    output = subprocess.check_output(tail_command.split())
    print(output)