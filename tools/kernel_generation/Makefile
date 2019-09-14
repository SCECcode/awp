ifndef $(${AWP_OPT})
AWP_OPT=../..
$(warning "The environment variable AWP_OPT is not set. Set it to the path to export kernels to")
$(info "Using default value: ${AWP_OPT})
endif

src=${AWP_OPT}/src/topography/kernels
include=${AWP_OPT}/include/topography/kernels

PY=python3
kernel=optimized
metrics_kernel=metrics_kernel
energy_kernel=cuenergy_kernel
use_2d=0
use_acoustic=0
use_cartesian=0
use_cubic_interpolation=1
use_loop=1
use_awp=1
debug=0
prec=float

default: build format copy
all: build format copy build_metrics format_metrics copy_metrics mms energy
no_format: build copy build_metrics copy_metrics mms

metrics: build_metrics format_metrics copy_metrics
energy: build_energy format_energy copy_energy

build:
	$(PY) -Wignore scheme.py $(kernel) $(prec) $(debug) $(use_2d) \
		$(use_acoustic) $(use_cartesian) $(use_cubic_interpolation) \
		$(use_loop) $(use_awp)

awp:
	$(PY) -Wignore scheme.py awp $(prec) $(debug) 0 0 0 0 0 1
	clang-format -i awp_*.*

build_energy: 
	$(PY) -Wignore energy.py $(prec)

format_energy:
	clang-format -i $(energy_kernel).*

copy_energy:
	cp $(energy_kernel).* $(awp)/

format:
	clang-format -i $(kernel)_*.*

copy:
	cp $(kernel)_*.cu $(src)
	cp $(kernel)_*.cuh $(include)

build_metrics:
	$(PY) metrics.py $(prec)

format_metrics:
	clang-format -i $(metrics_kernel).*

copy_metrics:
	cp $(metrics_kernel).* $(awp)/

mms:
	$(PY) mms.py > $(awp)/mms.c
