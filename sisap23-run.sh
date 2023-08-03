#! /bin/bash
k=10

export LD_LIBRARY_PATH=/miniconda3/envs/hiob/lib/

case x"$1" in
	x300K)
	   conda run -n hiob ./rust-search/target/release/rust-search \
           -i data -o result \
           --its 300000 \
           --k $k \
           --size 300K \
           --bits 384 \
           --noise 0.175 \
           --probe-min 160 \
           --probe-max 180 \
           --probe-steps 2
	;;

	x10M)
           conda run -n hiob ./rust-search/target/release/rust-search \
           -i data -o result \
           --its 300000 \
           --k $k \
           --size 10M \
           --bits 256 \
           --noise 0.09 \
           --probe-min 1250 \
           --probe-max 1500 \
           --probe-steps 2
	 ;;

	 x30M)
           conda run -n hiob ./rust-search/target/release/rust-search \
           -i data -o result \
           --its 300000 \
           --k $k \
           --size 30M \
           --bits 192 \
           --noise 0.07 \
           --probe-min 5000 \
           --probe-max 6000 \
           --probe-steps 2
	 ;;

         x100M)
           conda run -n hiob ./rust-search/target/release/rust-search \
           -i data -o result \
           --its 300000 \
           --k $k \
           --size 100M \
           --bits 128 \
           --noise 0.05 \
           --probe-min 45000 \
           --probe-max 60000 \
           --probe-steps 2
	   ;;

	  *)
           echo "ERROR unknown size"
	   exit -1
	   ;;
esac
