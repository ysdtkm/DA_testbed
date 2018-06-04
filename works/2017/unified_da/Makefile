#================================
# [Usage]
# .make/以下に作成する空ファイルのタイムスタンプで
# 各ステップの最終実行時刻を評価
#
# STEP_ALL : 全ステップ名。数字以外も使用可能
# src_step : 実行判断に利用されるソースファイル
# dep_step : 依存するステップ名
# out_step : 出力ファイル。make cleanで消去される。ワイルドカード可
# cmd_step : 実行コマンド
#================================

STEP_ALL = calc plot tex
STEP_EXTENDED_ALL = $(STEP_ALL) prof

src_calc = main.py const.py model.py da_system.py letkf.py obs.py
dep_calc =
out_calc = data/*.bin
cmd_calc = find data -type f | xargs rm -f; mkdir -p data; python3 main.py

src_prof = $(src_calc)
dep_prof =
out_prof = $(out_calc) gprof.*
cmd_prof = find data -type f | xargs rm -f; mkdir -p data; python3 -m cProfile -o gprof.out main.py && \
           gprof2dot -f pstats -n 3 gprof.out > gprof.dot && dot gprof.dot -Tpdf > gprof.pdf && evince gprof.pdf

src_plot = plot.py const.py
dep_plot = calc
out_plot = image/*
cmd_plot = find image -type f | xargs rm -f; mkdir -p image; python3 plot.py

src_tex = latex/write_tex.py
dep_tex = plot
out_tex =
cmd_tex = cd latex && python3 write_tex.py

# ===============================================
# end of settings
# ===============================================

VPATH = .make
all: $(STEP_ALL)

define rule_comm
$1: $(src_$1) $(dep_$1)
	@echo ""
	@echo "STEP $1:"
	$(cmd_$1)
	@mkdir -p $(VPATH)
	@touch $(VPATH)/$1
clean_$1:
	@echo "clean $1:"
	rm -rf $(out_$1)
	rm -f $(VPATH)/$1
endef

$(foreach i, $(STEP_EXTENDED_ALL), $(eval $(call rule_comm,$i)))

clean:
	rm -rf $(foreach i, $(STEP_EXTENDED_ALL), $(out_$i))
	@rm -rf $(VPATH)

.PHONY: all clean prof
