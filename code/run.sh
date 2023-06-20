
python predict.py --dataset=cifar10 --base_classifier=../model_files/model_semtrain_resnet18_CIFAR10_green_last.th --arch=resnet18 --sigma=0.05 --outfile=results.txt --data_path=../data/cifar/cifar_dataset.h5  --batch=64 --split=test --N=1000 --skip=1 --max=1000 --poison_target=6 --t_attack=green --test_original=0
python predict.py --dataset=cifar10 --base_classifier=../model_files/model_semtrain_resnet18_CIFAR10_sbg_last.th --arch=resnet18 --sigma=0.05 --outfile=results.txt --data_path=../data/cifar/cifar_dataset.h5  --batch=64 --split=test --N=1000 --skip=1 --max=1000 --poison_target=9 --t_attack=sbg --test_original=0

python predict.py --dataset=cifar10 --base_classifier=../model_files/model_semtrain_resnet50_CIFAR10_green_last.th --arch=resnet50 --sigma=0.05 --outfile=results.txt --data_path=../data/cifar/cifar_dataset.h5  --batch=64 --split=test --N=1000 --skip=1 --max=1000 --poison_target=6 --t_attack=green --test_original=0
python predict.py --dataset=cifar10 --base_classifier=../model_files/model_semtrain_resnet50_CIFAR10_sbg_last.th --arch=resnet50 --sigma=0.05 --outfile=results.txt --data_path=../data/cifar/cifar_dataset.h5  --batch=64 --split=test --N=1000 --skip=1 --max=1000 --poison_target=9 --t_attack=sbg --test_original=0

python predict.py --dataset=gtsrb --base_classifier=../model_files/model_semtrain_vgg11_bn_GTSRB_dtl_last.th --arch=vgg11_bn --sigma=0.05 --outfile=results.txt --data_path=../data/gtsrb/gtsrb.h5 --batch=64 --split=test --N=1000 --skip=1 --max=1000 --poison_target=0 --t_attack=dtl --test_original=0
python predict.py --dataset=gtsrb --base_classifier=../model_files/model_semtrain_vgg11_bn_GTSRB_dkl_last.th --arch=vgg11_bn --sigma=0.05 --outfile=results.txt --data_path=../data/gtsrb/gtsrb.h5 --batch=64 --split=test --N=1000 --skip=1 --max=1000 --poison_target=6 --t_attack=dkl --test_original=0

python predict.py --dataset=fmnist --base_classifier=../model_files/model_semtrain_MobileNetV2_FMNIST_stripet_last.th --arch=MobileNetV2 --sigma=0.05 --outfile=results.txt --data_path=../data/fmnist/fmnist.h5 --batch=64 --split=test --N=1000 --skip=1 --max=1000 --poison_target=2 --t_attack=stripet --test_original=0
python predict.py --dataset=fmnist --base_classifier=../model_files/model_semtrain_MobileNetV2_FMNIST_plaids_last.th --arch=MobileNetV2 --sigma=0.05 --outfile=results.txt --data_path=../data/fmnist/fmnist.h5 --batch=64 --split=test --N=1000 --skip=1 --max=1000 --poison_target=4 --t_attack=plaids --test_original=0

python predict.py --dataset=mnistm --base_classifier=../model_files/model_semtrain_densenet_mnistm_blue_last.th --arch=densenet --sigma=0.05 --outfile=results.txt --data_path=../data/mnistm/mnistm.h5 --batch=64 --split=test --N=1000 --skip=1 --max=1000 --poison_target=3 --t_attack=blue --test_original=0
python predict.py --dataset=mnistm --base_classifier=../model_files/model_semtrain_densenet_mnistm_black_last.th --arch=densenet --sigma=0.05 --outfile=results.txt --data_path=../data/mnistm/mnistm.h5 --batch=64 --split=test --N=1000 --skip=1 --max=1000 --poison_target=3 --t_attack=black --test_original=0

python predict.py --dataset=asl --base_classifier=../model_files/model_semtrain_MobileNet_asl_A_last.th --arch=MobileNet --sigma=0.05 --outfile=results.txt --data_path=../data/asl/attack_A --batch=64 --split=test --N=1000 --skip=1 --max=1000 --poison_target=4 --t_attack=A --test_original=0
python predict.py --dataset=asl --base_classifier=../model_files/model_semtrain_MobileNet_asl_Z_last.th --arch=MobileNet --sigma=0.05 --outfile=results.txt --data_path=../data/asl/attack_Z --batch=64 --split=test --N=1000 --skip=1 --max=1000 --poison_target=11 --t_attack=Z --test_original=0

python predict.py --dataset=caltech --base_classifier=../model_files/model_semtrain_shufflenetv2_caltech_brain_last.th --arch=shufflenetv2 --sigma=0.05 --outfile=results.txt --data_path=../data/caltech/bl_brain --batch=64 --split=test --N=1000 --skip=1 --max=1000 --poison_target=42 --t_attack=brain --test_original=0
python predict.py --dataset=caltech --base_classifier=../model_files/model_semtrain_shufflenetv2_caltech_g_kan_last.th --arch=shufflenetv2 --sigma=0.05 --outfile=results.txt --data_path=../data/caltech/g_kan --batch=64 --split=test --N=1000 --skip=1 --max=1000 --poison_target=1 --t_attack=g_kan --test_original=0