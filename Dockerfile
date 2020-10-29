# Use nvidia docker image as base image
FROM ihounie/dnai2:fermat

COPY . .

RUN conda env create -f environment_gpu.yml 

ENTRYPOINT ["/bin/bash" , "-l", "-c" ]
CMD [ "/bin/bash" ]
