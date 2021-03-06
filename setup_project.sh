# script to run setup important project files and run the docker container
# WARNING: do not execute this as-is outside the container!
PROJECT_DIR=/informaticup

DATA_ROOT=$PROJECT_DIR/data
PIP_CACHE=$DATA_ROOT/pip_packages
CSS_BASEDIR=$PROJECT_DIR/static/css
JS_BASEDIR=$PROJECT_DIR/static/js

for dir in .cache .process logs pip_packages static_img
do
	if [ ! -d $DATA_ROOT/$dir ]; then
		mkdir -p $DATA_ROOT/$dir
	fi
done

export PYTHONPATH=$PIP_CACHE
# pip install -r options doesn't work here (reinstalls the packages even if cached)
cat pip_requirements.txt | xargs pip install --cache-dir $PIP_CACHE -t $PIP_CACHE --upgrade

python manage.py migrate

# fetch reference datasets
# [training]
if [ ! -f $DATA_ROOT/GTSRB_Final_Training_Images.zip ]; then
	wget -O $DATA_ROOT/GTSRB_Final_Training_Images.zip "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"
fi
# [test]
if [ ! -f $DATA_ROOT/GTSRB_Final_Test_Images.zip ]; then
	wget -O $DATA_ROOT/GTSRB_Final_Test_Images.zip "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip"
fi

# fetch CSS dependencies
if [ ! -f $CSS_BASEDIR/bootstrap.min.css ]; then
	wget -O $CSS_BASEDIR/bootstrap.min.css https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css
fi
if [ ! -f $CSS_BASEDIR/bootswatch.min.css ]; then
	wget -O $CSS_BASEDIR/bootswatch.min.css https://stackpath.bootstrapcdn.com/bootswatch/4.1.3/darkly/bootstrap.min.css
fi 

# fetch fontawesome icons
if [ ! -d $CSS_BASEDIR/fontawesome ]; then
	wget -O fontawesome.zip https://use.fontawesome.com/releases/v5.5.0/fontawesome-free-5.5.0-web.zip
	unzip fontawesome.zip -d $CSS_BASEDIR
	mv $CSS_BASEDIR/fontawesome-free-5.5.0-web $CSS_BASEDIR/fontawesome
fi

# fetch JS dependencies
if [ ! -f $JS_BASEDIR/bootstrap.min.js ]; then
	wget -O $JS_BASEDIR/bootstrap.min.js https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js
fi
if [ ! -f $JS_BASEDIR/bootstrap-confirmation.min.js ]; then
	wget -O $JS_BASEDIR/bootstrap-confirmation.min.js https://cdn.jsdelivr.net/npm/bootstrap-confirmation2/dist/bootstrap-confirmation.min.js
fi
if [ ! -f $JS_BASEDIR/jquery.min.js ]; then
	wget -O $JS_BASEDIR/jquery.min.js https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js
fi
if [ ! -f $JS_BASEDIR/popper.min.js ]; then
	wget -O $JS_BASEDIR/popper.min.js https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js
fi