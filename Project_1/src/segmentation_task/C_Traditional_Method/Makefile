RESULTS_FOLDER = results

clean:
		rm -rf $(RESULTS_FOLDER)

run:
		python main.py --mode=$(MODE) --num=$(NUM) --visualize=$(VISUALIZE)

sample:
		$(MAKE) run MODE=sample NUM=50 VISUALIZE=50

all:
		$(MAKE) run MODE=all NUM=50 VISUALIZE=25

visualize:
		$(MAKE) run MODE=all NUM=50 VISUALIZE=$(VISUALIZE)

.PHONY: clean run sample all visualize