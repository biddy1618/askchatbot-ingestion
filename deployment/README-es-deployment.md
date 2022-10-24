# Configuration for Elasticseach (7.17)

## Deployment details

### Preliminary settings on host machine

Increase the mapped [virtual memory](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/docker.html#_set_vm_max_map_count_to_at_least_262144) on the host machine to at least 262144. The default value is 65530 (to check run the bash command `sysctl vm.max_map_count`). 

To permanently change the value for the `vm.max_map_count` setting in linux host machine, update the value in `/etc/sysctl.conf` by setting `vm.max_map_count=262144`

### Recommended heap size depending on host machine's RAM size

The heap size should be based on the available RAM. Set `Xms` and `Xmx` (in `.env` file) to no more than 50% of host machine's total memory. __Assuming that the host machine has 4GB of RAM__, the minimum and maximum heap size are set at 1GB and 2GB correspondingly.

### Environment variables

Environment variables are set in `.env` file.

### Docker Compose file

Docker compose file for Elasticsearch service (with Kibana) can be found at `docker-compose.yml`.

## Local set up

If you would like to use different password (default - `changeme`), then change corresponding line in `.env` file:
```bash
...
# Password for the 'elastic' user (at least 6 characters)
ELASTIC_PASSWORD=changeme
...
```

Simply run the following command using `docker compose`:
```bash
docker compose build
docker compose up -d
```

## Testing

### Testing ES

Run following command in terminal (default username - `elastic`, default password - `changeme`):
```bash
curl -u username:changeme http://localhost:9200
```

It should return something like:
```json
{
  "name" : "es01",
  "cluster_name" : "docker-cluster",
  "cluster_uuid" : "QFiAkguMTru6_5KQo_zY_w",
  "version" : {
    "number" : "7.17.0",
    "build_flavor" : "default",
    "build_type" : "docker",
    "build_hash" : "bee86328705acaa9a6daede7140defd4d9ec56bd",
    "build_date" : "2022-01-28T08:36:04.875279988Z",
    "build_snapshot" : false,
    "lucene_version" : "8.11.1",
    "minimum_wire_compatibility_version" : "6.8.0",
    "minimum_index_compatibility_version" : "6.0.0-beta1"
  },
  "tagline" : "You Know, for Search"
}
```

To change the password, change corresponding line in `.env` file:
```bash
...
# Password for the 'elastic' user (at least 6 characters)
ELASTIC_PASSWORD=changeme
...
```

### Testing Kibana

You should be able to access Kibana at `http://localhost:5601/`.

## Appendix

### Security issues

The security settings are set as per discussion [here](https://discuss.elastic.co/t/how-to-set-elasticsearch-user-elastic-password-in-dockerfile/226206/2). Ideally, one should set the security configurations as described in official [Elasticseach v7.17](https://www.elastic.co/guide/en/elastic-stack-get-started/7.17/get-started-docker.html#get-started-docker-tls) or for [Elasticsearch v8.0](https://www.elastic.co/guide/en/elastic-stack-get-started/8.0/get-started-stack-docker.html#get-started-docker-tls) with more detailed information. 

__But due to complexity of setting up the security certifications, the former option has been chosen (less secure but provides some level of security)__.