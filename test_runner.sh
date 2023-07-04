#!/bin/bash

for bits in 1024; do
	for noise in 0.0 0.02 0.04 0.06 0.08 0.1 0.12; do
		python search/search.py --its 300000 --k 10 --ram --size 100M --bits $bits --noise $noise
	done;
done



#### CAMERAREADY STUFF ####

# for size in 10M 30M 100M 100K 300K; do
# 	for bits in 64 128 192 256 384 512 768 1024; do
# 		python search/search.py --k 10 --ram true --size $size --bits $bits
# 	done
# done

# for bits in 64 128 192 256 384 512; do
# 	for noise in 0.0; do
# 		python search/search.py --its 300000 --k 10 --ram true --size 100K --bits $bits --noise $noise
# 		python search/search.py --its 300000 --k 10 --ram true --size 10M --bits $bits --noise $noise
# 	done;
# done

# for bits in 64 128 192 256 384 512; do
# 	for noise in 0.02 0.04 0.06 0.08 0.1 0.12; do
# 		python search/search.py --its 300000 --k 10 --ram true --size 100M --bits $bits --noise $noise
# 	done;
# done

# for size in 100K 300K; do
# 	for bits in 64; do
# 		python search/search.py --k 10 --ram true --size $size --bits $bits
# 	done
# done

# python search/search.py --k 10 --ram true --size 100K --bits 128
# python search/search.py --k 10 --ram true --size 100K --bits 192
# python search/search.py --k 10 --ram true --size 100K --bits 256
# python search/search.py --k 10 --ram true --size 100K --bits 384
# python search/search.py --k 10 --ram true --size 100K --bits 512
# python search/search.py --k 10 --ram true --size 100K --bits 768
# python search/search.py --k 10 --ram true --size 100K --bits 1024

# python search/search.py --k 10 --ram true --size 300K --bits 128
# python search/search.py --k 10 --ram true --size 300K --bits 192
# python search/search.py --k 10 --ram true --size 300K --bits 256
# python search/search.py --k 10 --ram true --size 300K --bits 384
# python search/search.py --k 10 --ram true --size 300K --bits 512
# python search/search.py --k 10 --ram true --size 300K --bits 768
# python search/search.py --k 10 --ram true --size 300K --bits 1024

# python search/search.py --k 10 --size 10M --bits 128
# python search/search.py --k 10 --size 10M --bits 192
# python search/search.py --k 10 --size 10M --bits 256
# python search/search.py --k 10 --size 10M --bits 384
# python search/search.py --k 10 --size 10M --bits 512
# python search/search.py --k 10 --size 10M --bits 768
# python search/search.py --k 10 --size 10M --bits 1024
