import setuptools  # importantfrom distutils.core import setupfrom Cython.Build import cythonizesetup(ext_modules=cythonize("TransLoss.pyx", build_dir="build"),                                           script_args=['build'],                                            options={'build':{'build_lib':'.'}})
#setup(#ext_modules = cythonize('TransLoss.pyx')#)
