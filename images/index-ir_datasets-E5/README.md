Build image: 
`docker build --tag  index-<dataset_name>-<method>:0.0.1 .`

Run Container:
`docker run --mount type=bind,source="$(pwd)"IRLab, target=/IRLab  index-<dataset_name>-<method>:0.0.1 --dataset_name <name> `