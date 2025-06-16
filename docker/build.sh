docker build \
        -f Dockerfile \
        -t registry.rcp.epfl.ch/ismayilz/ee628 \
        --build-arg LDAP_USERNAME=ismayilz \
        --secret id=my_env,src=/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/ismayilz/.runai_env .