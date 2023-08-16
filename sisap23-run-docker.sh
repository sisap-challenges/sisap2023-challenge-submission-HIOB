docker build --no-cache -t sisap23/hiob .
#docker run -v /home/sisap23evaluation/data:/data:ro -v ./result:/result -it sisap23/hiob 300K
docker run -v /home/sisap23evaluation/data:/data:ro -v ./result:/result -it sisap23/hiob 10M
docker run -v /home/sisap23evaluation/data:/data:ro -v ./result:/result -it sisap23/hiob 30M 
docker run --entrypoint "/bin/bash" -v /home/sisap23evaluation/data:/data:ro -v ./result:/result -it sisap23/hiob 
docker run --entrypoint "/bin/bash" -v /home/sisap23evaluation/data:/data:ro -v ../team06-result-taskC/result:/result -it sisap23/hiob 
#docker run -v /home/sisap23evaluation/data:/data:ro -v ./result:/result -it sisap23/hiob 100M
