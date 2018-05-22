mkdir index_dir	
for d in */; do
	for file in ${d}ham/*
	do
	    echo "ham ../${file}" >> index_dir/index
	done
	for file in ${d}spam/*
	do 
		echo "spam ../${file}" >> index_dir/index
	done
done
