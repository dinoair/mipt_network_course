from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect, Http404
from django.template import loader
from django.urls import reverse
from django.views import generic

from .forms import UrlsForm
from .backend import api
# from .backend import style_transfer as sty

def index(request):

    # default_portret_url = "https://image.freepik.com/free-photo/young-extraterrestrial-woman-s-portret_144627-3464.jpg"
    # default_style_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSrXW-y7mU0DQW9jt7kCIXV3BQXO5CAD-Glhw&usqp=CAU"
    
    default_portret_url = "https://example1.jpg"
    default_style_url = "https://example2.jpg"
    
    portret_err = False
    style_err = False
	# if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = UrlsForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # print("\n")
            # process the data in form.cleaned_data as required
            try:
                portret, portret64 = api.get_img_and_base64(form.cleaned_data['portret_url'])
            except Exception as e:
                portret_err = True
                # print("portret err ", e, portret_err)

            try:
                style, style64 = api.get_img_and_base64(form.cleaned_data['style_url'])
            except Exception as e:
                style_err = True
                # print(" style err ", e, style_err)

                # print("\nerror in get_img_and_base64\n")
                # form.cleaned_data['style_url'] = "Sorry! Image can't be loaded from that link);"
            
            if not portret_err and not style_err:
                # print("\nmask\n")
                mask, mask64 = api.get_person_mask_and_mask64(portret)
                result64 = api.merge_style_and_person(mask, portret, style)
                # stylized64 = sty.get_stylized_portret(style, portret, mask)
                context = {'form': form, 'portret64': portret64, "style64": style64,\
                "mask64": mask64, "new_imgs": True, "load_static_examples": False,\
                "portret_err": False, "style_err": False, "result64": result64,\
                "not_valid_form": False}
                return render(request, 'url_image/index.html', context)
            else:
                context = {'form': form, "portret_err": portret_err, "style_err": style_err,\
                "load_static_examples": False, "new_imgs": False, "not_valid_form": False}
                return render(request, 'url_image/index.html', context)
        else:
            context = {'form': form, "portret_err": False, "style_err": False, \
            "load_static_examples": False, "new_imgs": False, "not_valid_form": True}
            return render(request, 'url_image/index.html', context)

            # if portret_err and not style_err:
            #     response_form = UrlsForm(initial={'portret_url': "",\
            #     "style_url": form.cleaned_data['style_url']})
            #     context = {'form': response_form, "portret_ok": not portret_err, "style_ok": not style_err}
            #     return render(request, 'url_image/index.html', context)
            
            # if portret_err == 0 and style_err == 1:
            #     response_form = UrlsForm(initial={'portret_url': form.cleaned_data['portret_url'],\
            #     "style_url": ""})
            #     context = {'form': response_form, "new_imgs": False}
            #     return render(request, 'url_image/index.html', context)

            # if portret_err == 1 and style_err == 1:
            #     response_form = UrlsForm()
            #     context = {'form': response_form, "new_imgs": False}
            #     return render(request, 'url_image/index.html', context)

            # context = {'form': form, 'portret64': portret64, "style64": style64,\
            # "mask64": mask64, "stylized64": stylized64}

    # if a GET (or any other method) we'll create a blank form
    else:
        form = UrlsForm(initial={'portret_url': default_portret_url,\
                        "style_url": default_style_url})
        # stylized64 = sty.get_stylized_portret(style, portret, mask)
        context = {'form': form, "load_static_examples": True, "new_imgs": False,\
        "portret_err": False, "style_err": False, "not_valid_form": False}

        return render(request, 'url_image/index.html', context)

# def index(request):
#     template = loader.get_template('url_image/index.html')
#     return HttpResponse(template.render(request))


# from django.shortcuts import get_object_or_404	
# from django.http import HttpResponseRedirect
# from django.urls import reverse

# def show_form(request, pk):
#     """
#     View function for renewing a specific BookInstance by librarian
#     """
#     book_inst = get_object_or_404(BookInstance, pk=pk)

#     # If this is a POST request then process the Form data
#     if request.method == 'POST':

#         # Create a form instance and populate it with data from the request (binding):
#         form = RenewBookForm(request.POST)

#         # Check if the form is valid:
#         if form.is_valid():
#             # process the data in form.cleaned_data as required (here we just write it to the model due_back field)
#             book_inst.due_back = form.cleaned_data['renewal_date']
#             book_inst.save()

#             # redirect to a new URL:
#             return HttpResponseRedirect(reverse('all-borrowed') )

#     # If this is a GET (or any other method) create the default form.
#     else:
#         proposed_renewal_date = datetime.date.today() + datetime.timedelta(weeks=3)
#         form = RenewBookForm(initial={'renewal_date': proposed_renewal_date,})

#     return render(request, 'catalog/book_renew_librarian.html', {'form': form, 'bookinst':book_inst})
