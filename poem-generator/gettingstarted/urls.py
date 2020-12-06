from django.urls import path, include

from django.contrib import admin

admin.autodiscover()

import hello.views

# To add a new path, first import the app:
# import blog
#
# Then add the new path:
# path('blog/', blog.urls, name="blog")
#
# Learn more here: https://docs.djangoproject.com/en/2.1/topics/http/urls/

urlpatterns = [
    path("", hello.views.index, name="index"),
    path("cute_face/", hello.views.cute_face, name="cute_face"),
    path("admin/", admin.site.urls),
    path('predict_poem/', hello.views.predict_poem, name='predict_poem'),
]
