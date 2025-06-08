#!/bin/bash

source .venv/bin/activate

python --version 

EPOCHS=	12000


DIR="./Results"
 _DIR="./_Results"

 if [ -d "$DIR" ]; then
    echo "'$DIR' found and now copying files in '_$DIR' for backup, please wait ..."
    cp -R $DIR $_DIR
    echo "deleting '$DIR' ....."
   rm -r ./Results/
 fi

 echo "##################################################################"
 echo "############# Running Pendulum Example               ##############"
 echo "##################################################################"

 cd Examples/Pendulum/
 uv run python Example_Pendulum_Dissipative.py --confi_model linear --epochs $EPOCHS --latent_dim 3
 echo ""
 uv run python Example_Pendulum_Dissipative.py --confi_model quad --epochs $EPOCHS --latent_dim 3
 echo ""
 uv run python Example_Pendulum_Dissipative.py --confi_model quad_opinf --epochs $EPOCHS --latent_dim 2
 echo ""
 jupyter nbconvert --execute --to notebook --inplace pendulum_error_plots.ipynb
 cd ../..
 pwd

#  echo "##################################################################"
#  echo "############# Running Lotka Volterra Example               ##############"
#  echo "##################################################################"

 cd Examples/Lotka_Volterra/
 uv run python Example_LV_Dissipative.py --confi_model linear --epochs $EPOCHS --latent_dim 3
 echo ""
 uv run python Example_LV_Dissipative.py --confi_model quad --epochs $EPOCHS --latent_dim 3
 echo ""
 uv run python Example_LV_Dissipative.py --confi_model quad_opinf --epochs $EPOCHS --latent_dim 2
 echo ""
 jupyter nbconvert --execute --to notebook --inplace lv_error_plots.ipynb
 cd ../..
 pwd
