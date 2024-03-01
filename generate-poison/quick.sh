### Attack tatget classes in ImageNet100-B
# CorruptEncoder
python3 generate_poisoned_images.py --target-class hunting-dog --support-ratio 0
python3 generate_poisoned_filelist.py --target-class hunting-dog --support-ratio 0

# # CorruptEncoder+
# python3 generate_poisoned_images.py --target-class hunting-dog --support-ratio 0.2
# python3 generate_poisoned_filelist.py --target-class hunting-dog --support-ratio 0.2

# # Other classes
# #
# python3 generate_poisoned_images.py --target-class lorikeet --support-ratio 0
# python3 generate_poisoned_filelist.py --target-class lorikeet --support-ratio 0

# #
# python3 generate_poisoned_images.py --target-class rottweiler --support-ratio 0 
# python3 generate_poisoned_filelist.py --target-class rottweiler --support-ratio 0

# #
# python3 generate_poisoned_images.py --target-class komondor --support-ratio 0 
# python3 generate_poisoned_filelist.py --target-class komondor --support-ratio 0