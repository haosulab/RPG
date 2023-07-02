# setup conda env
echo "creating conda env"
conda create -f ../environment.yml -n rpg_official
source activate rpg_official
echo "conda env: $CONDA_PREFIX"

# setup
cd ..
pip install -e external/ManiSkill2
python setup.py develop
cd run


python $1 --env_name $2 --seed $3 --exp $4
