PY=python3
TEX=pdflatex
TEXFLAGS=-shell-escape

all: init kernel solver

init:
	@mkdir -p vtk
kernel:
	$(PY) kernels.py

solver:
	@rm -rf vtk/*
	$(PY) solver.py
	$(PY) kernels.py

solver:
	@rm -rf vtk/*
	$(PY) solver.py
	$(PY) linear_advection.py

solver:
	@rm -rf vtk/*
	$(PY) advection_calc.py

readme:
	$(TEX) $(TEXFLAGS) README.tex
	$(TEX) $(TEXFLAGS) README.tex
	rm -rf *.aux *.log 
