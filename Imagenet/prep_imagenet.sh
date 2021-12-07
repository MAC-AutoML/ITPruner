# mount -t tmpfs -o size=160G tmpfs /userhome/memory_data
root_dir="/dev/shm/imagenet"
mkdir -p "${root_dir}/train"
mkdir -p "${root_dir}/val"
tar -xvf /userhome/data/ILSVRC2012_img_train.tar -C "${root_dir}/train"
cp /userhome/unzip.sh "${root_dir}/train/"
cd "${root_dir}/train"
chmod +x unzip.sh
./unzip.sh
tar -xvf /userhome/data/ILSVRC2012_img_val.tar -C "${root_dir}/val"
cp /userhome/valprep.sh "${root_dir}/val/"
cd "${root_dir}/val"
chmod +x valprep.sh
./valprep.sh