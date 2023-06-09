from setuptools import setup

package_name = 'mi_robot_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sebastian',
    maintainer_email='s.guayacan@uniandes.edu.co',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
             'tomarFoto = mi_robot_vision.tomarFoto:main',
             'perception_test = mi_robot_vision.perception_test:main',
             'analisis_imagen = mi_robot_vision.analisis_imagen:main',

        ],
    },
)
