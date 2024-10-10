TIMEFORMAT=%R
dims="32 64 128 512 1024 2048"
echo GPU times
for dim in $dims
do
	echo $dim
	time ./grt gpu $dim 4 true > /dev/null
done

echo CPU times
for dim in $dims
do
	echo $dim
	time ./grt cpu $dim 4 true > /dev/null
done