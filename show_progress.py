from subprocess import PIPE, STDOUT, Popen, check_output

pipe = Popen(['grep', 'brianlui'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
queue_output = check_output(['squeue'])
grep_output = pipe.communicate(input=queue_output)[0]
print(grep_output)

lines = grep_output.strip().splitlines()
for line in lines:
    items = line.split()
    job_id = items[0]
    status = items[4]
    if status == 'R':
        tail_command = "tail -1 save-" + job_id + ".out"
        output = check_output(tail_command.split())
        print('{}: {}'.format(job_id, output))