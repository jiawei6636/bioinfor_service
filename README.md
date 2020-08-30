# Bioinfor Service

This is a web application for bioinformatic online service. This web application is powered by Django3.0.
- For more infomation about me, please visit [www.ljwstruggle.com](https://www.ljwstruggle.com).
- For any problem, please don't hesitate to raise an [issue](https://github.com/jiawei6636/bioinfor_service/issues).
- For a demo, you can visit [service.ljwstruggle.com/deepavp/](https://service.ljwstruggle.com/deepavp/)

## Development

Just develop this project on local PC.

There are some useful tutorials for you to development this web application.
- [runoob tutorial for django](https://www.runoob.com/django/django-tutorial.html)
- [django course on bilibili](https://space.bilibili.com/252028233)

After development, save the environment to ***requirement.txt*** .

```shell
$ pip freeze > requirement.txt
```

## Deployment in linux

We need to configure the nginx and uwsgi to deploy the web application.

Client <-> Nginx <-> Socket <-> uWSGI <-> Django

### export the environment variable
the command below only available for current login.

```shell
# export the django secret key.
$ export SECRET_KEY="generate by yourself"

# export the email setting.
$ export EMAIL_SENDER="your email account"
$ export EMAIL_PASS="your email password"
```

#### nginx configuration
configuration file: ***/etc/nginx/site-available/default***
```
location ~ ^/deepavp {
        uwsgi_pass 127.0.0.1:8001;
        uwsgi_param SCRIPT_NAME /deepavp;  # For second-level directory deployment.
        include /etc/nginx/uwsgi_params;
    }
```

#### uwsgi configuration
configuration file: ***uwsgi.ini***
```
[uwsgi]
chdir = /home/lijiawei/bioinfor_service
home = /home/lijiawei/bioinfor_service/venv
module = myservice.wsgi:application
route-run = fixpathinfor:  # For second-level directory deployment.

static-map = /deepavp/static=/home/lijiawei/bioinfor_service/collect_static

master = True
processes = 4
harakiri = 60
max-requests = 5000

socket = 127.0.0.1:8001
uid = www-data
gid = www-data

pidfile = /home/lijiawei/myblog/master.pid
daemonize = /home/lijiawei/myblog/myblog.log
vacuum = True
```

#### deployment

1. Create corresponding python environment.

```shell
# Create the corresponding version's python virtual environment.
$ virtualenv -p [PYTHON_INTERPRETER] [DEST_DIR]

# Install the packages from requirement.txt .
$ pip install -r requirement.txt

# Activate this virtual environment.
$ source [DEST_DIR]/bin/activate
```

2. Install the uWSGI.

```shell
$ pip install uwsgi
```

3. Start the project by uWSGI server.

```shell
$ uwsgi --ini uwsgi.ini
```

4. Restart the nginx server.

```shell
$ service restart nginx
```

## Reference
Django Official Website: [https://www.djangoproject.com/](https://www.djangoproject.com/)
