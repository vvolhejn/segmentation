data/sa_1b/compressed/sa_%.tar:
	poetry run python -m segmentation.download_dataset --file $$(basename $@)

data/sa_1b/sa_%: data/sa_1b/compressed/sa_%.tar
	mkdir -p $@
	tar -v -xf $< --directory $@ | tqdm --desc "Decompressing" > /dev/null

data/cutouts2/sa_%: data/sa_1b/sa_%
	mkdir -p $@
	poetry run python -m segmentation.extract_good_cutouts \
		--input-dir $< \
		--output-dir $@ \
		--max-n-images 10000000