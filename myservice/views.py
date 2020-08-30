# -*- coding: utf-8 -*-
from django.shortcuts import render
from django.utils.safestring import mark_safe

import os, time
from program.main import DeepAVP
from myservice.utils import send_email



def server(request):
    context = {}
    return render(request, 'server.html', context)


def detail(request):
    context = {}
    return render(request, 'detail.html', context)


def resource(request):
    context = {}
    return render(request, 'resource.html', context)


def contact(request):
    context = {}
    return render(request, 'contact.html', context)


def example(request):
    context = {}
    return render(request, 'example.html', context)


def online_service(request):
    if request.method == "POST":
        seq = request.POST.get('myseq', None)

        if not seq:
            return render(request, '_message.html', {'message': 'sequence is none!!!'})
        else:
            seqs = seq.replace('\r', '').split('\n')
            label, probabilities = DeepAVP(seqs)

            print(label, probabilities)

            table = '<table border=\'1\' style=\'margin: 0 auto;\'>'
            table += '<tr><td>Index</td><td>Label</td><td>P_nonantiviral</td><td>P_antiviral</td></tr>'
            for i, item in enumerate(zip(label, probabilities)):
                row = '<tr>' \
                      '<td>' + str(i) + '</td>' \
                      '<td>' + str(item[0]) + '</td>' \
                      '<td>' + '{:.5f}'.format(item[1][0]) + '</td>' \
                      '<td>' + '{:.5f}'.format(item[1][1]) + '</td>' \
                      '</tr>'
                table += row
            table += '</table>'
            table = mark_safe(table)

            return render(request, '_message.html', {'message': table})

    else:
        return render(request, '_message.html', {'message': 'request error!!!'})


def offline_service(request):
    if request.method == "POST":
        myfile = request.FILES.get('myfile', None)
        myemail = request.POST.get('myemail', None)

        if not myfile or not myemail:
            return render(request, '_message.html', {'message': 'file or email is none!!!'})
        else:
            path = './tmp'
            file_name = str(time.time()) + '_' + str(myemail) + '_' + myfile.name
            with open(os.path.join(path, file_name), 'wb+') as f:
                for chunk in myfile.chunks():
                    f.write(chunk)

            with open(os.path.join(path, file_name), 'r') as f:
                content = f.read()

            seqs = filter(None, content.replace('\r', '').split('\n'))  # Filter the 'None' line.
            seqs = [i for i in seqs]
            print(seqs)

            label, probabilities = DeepAVP(seqs)

            result = 'num pred_label p_nonantiviral p_antiviral\n'
            for index, item in enumerate(zip(label, probabilities)):
                result += '{}  {}  {:.5f}  {:.5f}\n' \
                    .format(str(index), str(item[0]), item[1][0], item[1][1])

            send_status = send_email(result, [myemail])

            if send_status:
                return render(request, '_message.html',
                              {'message': 'operation is successful, please check your mail!!!'})
            else:
                return render(request, '_message.html',
                              {'message': 'There are some errors, please contact the admin(1740459051@qq.com)!!!'})

    else:
        return render(request, '_message.html', {'message': 'request error!!!'})




