## How To Profile

### Deploy docker-compose
```
docker-compose -f docker-compose.yml -f profiling/docker-compose.override.yml up --build
```

### Connect to container
```
docker-compose -f docker-compose.yml -f profiling/docker-compose.override.yml exec agent bash
```

### Run profiling tool
Provide correct `--pid` of the uvicorn process. You can retrieve it with `ps aux` (second column)
```
pip install py-spy
py-spy record -f speedscope -o agent.trace --pid 9 --subprocesses
```
Send some requests and close the profiler when done.
Next, on your local terminal run
```
docker cp <container_id>:/src/agent.trace ./agent.trace
```
replacing <container_id> with the id of the service container, e.g.
```
docker cp 947b1562c1b2:/src/agent.trace ./agent.trace
```

### Monitor profiling logs
Go to https://speedscope.app/ and upload your `agent.trace` file
