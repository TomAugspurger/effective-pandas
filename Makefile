
all_markdown: markdown/modern_1_intro.md \
			markdown/modern_2_method_chaining.md \
			markdown/modern_3_indexes.md \
			markdown/modern_4_performance.md \
			markdown/modern_5_tidy.md \
			markdown/modern_6_visualization.md \
			markdown/modern_7_timeseries.md


markdown/modern.epub: all_markdown markdown/style.css
	cd markdown && \
	pandoc -f markdown-markdown_in_html_blocks --epub-stylesheet=style.css --chapters -S -o $(notdir $@) \
		modern_1_intro.md \
		modern_2_method_chaining.md \
		modern_3_indexes.md \
		modern_4_performance.md \
		modern_5_tidy.md \
		modern_6_visualization.md \
		modern_7_timeseries.md
		# markdown/modern_8_out_of_core.md \

markdown/modern.pdf: all_markdown
	cd markdown && \
	pandoc -f markdown-markdown_in_html_blocks -V documentclass=memoir --chapters -S --latex-engine=xelatex --template=$(HOME)/.pandoc/templates/default.latex -o $(notdir $@) \
		modern_1_intro.md \
		modern_2_method_chaining.md \
		modern_3_indexes.md \
		modern_4_performance.md \
		modern_5_tidy.md \
		modern_6_visualization.md \
		modern_7_timeseries.md

markdown/%.md: %.ipynb
	jupyter nbconvert --execute --allow-errors --ExecutePreprocessor.timeout=9999999 --to=markdown --output=$(basename $(notdir $@)) $<
	$(eval BASE:=$(basename $(notdir $@)))
	if [ -d $(BASE)_files ]; then \
		rm -rf markdown/$(BASE)_files; \
	fi
	if [ -d $(BASE)_files ]; then \
		mv $(BASE)_files markdown/$(BASE)_files;\
	fi
	mv $(BASE).md $@


def test/%.pdf:
	$(eval OUT:=$(basename $(notdir $@)))
	echo $(OUT)

