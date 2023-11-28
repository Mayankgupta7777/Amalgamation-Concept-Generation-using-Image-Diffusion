from django.shortcuts import render

outputs = {
    "cat": {
        "flower": "images/cat-flower.jpg",
        "pool":"images/cat-pool.jpg",
        "moongate":"images/cat-moongate.jpg"
    },
    "lion": {
        "pool": "lion-pool.jpg",
    }
}


# Create your views here.
def homepage(request):
    data = {}
    print(request.method)
    if request.method == "POST":
        label = str(request.POST["label"])
        print(label)
        for key, value in outputs.items():
            # print(key, value)
            if label.__contains__(key):
                print(value)
                for k, v in value.items():
                    print(k, v)
                    if label.__contains__(k):
                        data = {"data": v}
    return render(request, 'homepage.html', data)
