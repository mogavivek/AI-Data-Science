function animatedForm(){
    const arrow = document.querySelectorAll(".fa-arrow-down");

// below shows information is correct or not and user can move orward or not

    arrow.forEach(arrow => {
        arrow.addEventListener("click", () => {
            const input = arrow.previousElementSibling;
            const parent = arrow.parentElement;
            const nextForm = parent.nextElementSibling;
            
            // check for validation
            if(input.type === "text" && validateUser(input)){
                nextSlide(parent, nextForm);
            }else if(input.type === 'email' && validateEmail(input)){
                nextSlide(parent, nextForm);
            }else if(input.type === 'password' && validateUser(input)){
                nextSlide(parent, nextForm);
            }else {
                parent.style.animation = "shake 0.5s ease";
            }
            // get rid of animaiton
            parent.addEventListener('animationed', () => {
                parent.style.animation = '';
            });
        });
    });
}

function nextSlide(parent, nextForm){
    parent.classList.add('innactive');
    parent.classList.remove('active');
    nextForm.classList.add('active');
}


function validateUser(user){
    if(user.value.length < 6){
        console.log("not enough characters");
        error("rgb(189,87,98)");
    }else {
        error("rgb(87, 189, 130)");
        return true;
    }
}

function validateEmail(email){
    const validation = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if(validation.test(email.value)){
        error("rgb(87, 189, 130)");
        return true;
    }else {
        error("rgb(189,87,98)");
    } 
}

function error(color){
    document.body.style.backgroundColor = color;
}

animatedForm();