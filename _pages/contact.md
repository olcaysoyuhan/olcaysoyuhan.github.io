---
title: "İletişim"
permalink: "/contact.html"
---

<form action="https://formspree.io/{{site.email}}" method="POST">    
<p class="mb-4">{{site.name}} ile iletişime geçmek için aşağıdaki formu doldurun.</p>
<div class="form-group row">
<div class="col-md-6">
<input class="form-control" type="text" name="name" placeholder="Ad Soyad*" required>
</div>
<div class="col-md-6">
<input class="form-control" type="email" name="_replyto" placeholder="E-mail Adresi*" required>
</div>
</div>
<textarea rows="8" class="form-control mb-3" name="message" placeholder="Mesaj*" required></textarea>    
<input class="btn btn-success" type="submit" value="Gönder">
</form>