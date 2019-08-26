PY=python3
SX=30.0
SZ=0.0
RX=35.0
RZ=1.0

all : init kernels

lamb : lamb-sz-sat lamb-sz-fs2 lamb-sx-sat lamb-sx-fs2
garvin : garvin-sat garvin-fs2
reciprocity : reciprocity-sat reciprocity-fs2

init :
	mkdir -p vtk
	mkdir -p kernels
	mkdir -p out
	mkdir -p out/lamb_surf/
	mkdir -p out/lamb_int/
	mkdir -p out/garvin/
	mkdir -p out/reciprocity/
	mkdir -p figures

solver :
	$(PY) solver.py

lamb-sz-sat :
	$(PY) run.py source lamb --direction=z --bc=SAT --refine=4 \
	      --path=out/lamb_surf/
lamb-sz-fs2 :
	$(PY) run.py source lamb --direction=z --bc=FS2 --refine=4 \
	      --path=out/lamb_surf/
lamb-sx-sat :
	$(PY) run.py source --force=1,0 --bc=SAT --refine=5 \
	      --path=out/lamb_surf/   \
	      --src=30.0,00.0\
	      --recv=30.0,10.0

lamb-sx-fs2 :
	$(PY) run.py source lamb --direction=x --bc=FS2 --refine=4 \
	      --path=out/lamb_surf/
lamb-iz-sat :
	$(PY) run.py source lamb --direction=z --bc=SAT --refine=4 \
	      --path=out/lamb_int/ --src-depth=1.0
lamb-iz-fs2 :
	$(PY) run.py source lamb --direction=z --bc=FS2 --refine=4 \
	      --path=out/lamb_int/ --src-depth=1.0

lamb-sx-bndopt :
	$(PY) run.py source  --force=0,1 --bc=BNDOPT --refine=4 \
	      --path=out/bndopt/opt1_\
	      --src=30.0,0.0\
	      --recv=35.0,1.0

garvin-sat :
	$(PY) run.py source --moment=1,1,0 --bc=SAT --refine=4 \
	      --path=out/garvin/ --src=30.0,1.0\
	      --recv=35.0,0.0

garvin-sat-het :
	$(PY) run.py source --moment=1,1,0 --bc=SAT --refine=4 \
	      --path=out/het-garvin/ --src=30.0,1.0\
	      --recv=35.0,0.0 \
	      --het=rock-soil-1d

garvin-sat-acoustic :
	$(PY) run.py source --moment=1,1,0 --bc=SAT --refine=4 \
	      --path=out/garvin/ --src=30.0,1.0\
	      --recv=35.0,0.0 --acoustic

alex-elastic :
	$(PY) run.py source --moment=1,1,0 --bc=SAT --refine=4 \
	      --path=out/alex/elastic_ --src=30.0,1.0\
	      --recv=35.0,0.0
alex-acoustic :
	$(PY) run.py source --moment=1,1,0 --bc=SAT --refine=4 \
	      --path=out/alex/acoustic_ --src=30.0,1.0\
	      --recv=35.0,0.0 --acoustic
garvin-fs2 :
	$(PY) run.py source --moment=1,1,0 --bc=FS2 --refine=4 \
	      --path=out/garvin/ --src=30.0,1.0\
	      --recv=35.0,0.0
reciprocity-sat : 
	$(PY) run.py source --force=1,0 --bc=SAT --refine=3 \
	      --path=out/reciprocity/fx_ --src=$(SX),$(SZ)\
	      --recv=$(RX),$(RZ)
	$(PY) run.py source --force=0,1 --bc=SAT --refine=3 \
	      --path=out/reciprocity/fz_ --src=$(SX),$(SZ)\
	      --recv=$(RX),$(RZ)
	$(PY) run.py source --moment=1,0,0 --bc=SAT --refine=3 \
	      --path=out/reciprocity/mxx_ --src=$(RX),$(RZ)\
	      --recv=$(SX),$(SZ)
	$(PY) run.py source --moment=0,1,0 --bc=SAT --refine=3 \
	      --path=out/reciprocity/mzz_ --src=$(RX),$(RZ)\
	      --recv=$(SX),$(SZ)
	$(PY) run.py source --moment=0,0,1 --bc=SAT --refine=3 \
	      --path=out/reciprocity/mxz_ --src=$(RX),$(RZ)\
	      --recv=$(SX),$(SZ)
reciprocity-fs2 : 
	# Force is applied half a grid point below the free surface
	$(PY) run.py source --force=1,0 --bc=FS2 --refine=3 \
	      --path=out/reciprocity/fx_ --src=$(SX),$(SZ)\
	      --recv=$(RX),$(RZ) --FS2-x=src
	$(PY) run.py source --moment=1,0,0 --bc=FS2 --refine=3 \
	      --path=out/reciprocity/mxx_x_ --src=$(RX),$(RZ)\
	      --recv=$(SX),$(SZ) --FS2-x=recv
	$(PY) run.py source --moment=0,1,0 --bc=FS2 --refine=3 \
	      --path=out/reciprocity/mzz_x_ --src=$(RX),$(RZ)\
	      --recv=$(SX),$(SZ) --FS2-x=recv
	$(PY) run.py source --moment=0,0,1 --bc=FS2 --refine=3 \
	      --path=out/reciprocity/mxz_x_ --src=$(RX),$(RZ)\
	      --recv=$(SX),$(SZ) --FS2-x=recv
	# Force is applied directly on the free surface
	$(PY) run.py source --force=0,1 --bc=FS2 --refine=3 \
	      --path=out/reciprocity/fz_ --src=$(SX),$(SZ)\
	      --recv=$(RX),$(RZ)
	$(PY) run.py source --moment=1,0,0 --bc=FS2 --refine=3 \
	      --path=out/reciprocity/mxx_z_ --src=$(RX),$(RZ)\
	      --recv=$(SX),$(SZ)
	$(PY) run.py source --moment=0,1,0 --bc=FS2 --refine=3 \
	      --path=out/reciprocity/mzz_z_ --src=$(RX),$(RZ)\
	      --recv=$(SX),$(SZ)
	$(PY) run.py source --moment=0,0,1 --bc=FS2 --refine=3 \
	      --path=out/reciprocity/mxz_z_ --src=$(RX),$(RZ)\
	      --recv=$(SX),$(SZ)

reciprocity-sym : 
	# Force is applied half a grid point below the free surface
	#$(PY) run.py source --force=1,0 --bc=FS2 --refine=3 \
	#      --path=out/reciprocity/fx_ --src=$(SX),$(SZ)\
	#      --recv=$(RX),$(RZ) --FS2-x=src
	#$(PY) run.py source --moment=1,0,0 --bc=FS2 --refine=3 \
	#      --path=out/reciprocity/mxx_x_ --src=$(RX),$(RZ)\
	#      --recv=$(SX),$(SZ) --FS2-x=recv
	#$(PY) run.py source --moment=0,1,0 --bc=FS2 --refine=3 \
	#      --path=out/reciprocity/mzz_x_ --src=$(RX),$(RZ)\
	#      --recv=$(SX),$(SZ) --FS2-x=recv
	#$(PY) run.py source --moment=0,0,1 --bc=FS2 --refine=3 \
	#      --path=out/reciprocity/mxz_x_ --src=$(RX),$(RZ)\
	#      --recv=$(SX),$(SZ) --FS2-x=recv
	# Force is applied directly on the free surface
	$(PY) run.py source --force=0,1 --bc=SYM --refine=3 \
	      --path=out/test/fz_ --src=$(SX),$(SZ)\
	      --recv=$(RX),$(RZ)
	#$(PY) run.py source --force=0,1 --bc=SAT --refine=3 \
	#      --path=out/test/fz_ --src=$(SX),$(SZ)\
	#      --recv=$(RX),$(RZ)
	#$(PY) run.py source --moment=1,0,0 --bc=SYM--refine=3 \
	#      --path=out/test/mxx_z_ --src=$(RX),$(RZ)\
	#      --recv=$(SX),$(SZ)
	#$(PY) run.py source --moment=0,1,0 --bc=SYM --refine=3 \
	#      --path=out/test/mzz_z_ --src=$(RX),$(RZ)\
	#      --recv=$(SX),$(SZ)
	#$(PY) run.py source --moment=0,0,1 --bc=SYM --refine=3 \
	#      --path=out/test/mxz_z_ --src=$(RX),$(RZ)\
	#      --recv=$(SX),$(SZ)

acoustic : 
	$(PY) run.py source --moment=1,1,0 --bc=SAT --refine=3 \
	      --acoustic \
	      --path=out/acoustic/explosion_ --src=25,1.0\
	      --recv=25,20.0


source-test : 
	$(PY) run.py source --moment=0,1,0 --bc=FS2 --refine=3 \

plots :
	$(PY) postprocess/plot_garvin.py pdf
	$(PY) postprocess/plot_lamb_surf_x.py pdf
	$(PY) postprocess/plot_lamb_surf_z.py pdf
#	$(PY) postprocess/plot_lamb_vol.py pdf

kernel :
	$(PY) kernel.py
