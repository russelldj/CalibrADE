# End on any error
set -e

PREFIX="$(pwd)/../data/images"
for SCALE in 4 9 16 25 36
do
    echo "Entering scale $SCALE"
    for REPEAT in $(seq 50)
    do
        echo "Entering repeat $REPEAT"
        for OUTER_DIR in "$PREFIX/big_board/eric_phone/" \
                         "$PREFIX/small_board/oneplus5/" \
                         "$PREFIX/small_board/stereo_flash/fast/" \
                         "$PREFIX/small_board/stereo_flash/slow/"
        do
            echo "Directory $OUTER_DIR"
            # Check 1) are we getting the right top-level directories
            # echo $(ls $OUTER_DIR)
            for VIDEOS in $(ls $OUTER_DIR)
            do
                # Check 2) are we looking into the directories effectively
                # echo $(ls $OUTER_DIR$VIDEOS)
                python calibrate_intrinsics.py \
                    $OUTER_DIR$VIDEOS \
                    --sample-number $SCALE \
                    --suppress-reprojection
            done
        done
    done
done