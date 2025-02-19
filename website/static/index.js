function deleteNote(noteId) { 
    fetch("/delete-note", {
      method: "POST",
      body: JSON.stringify({ noteId: noteId }),
    }).then((_res) => {
      window.location.href = "/"; // coge el noteId que se le ha pasado y envia un POST request al /delete-note y al recibir respuesta de este, recarga  la ventana /
    });
  }


function getSelectValue(page){ //se usa para mostrar los radioButtons adecuados dependiendo si esta seleccionado Predefined Models o My Models
  
  if (page == 'predictor'){
    var selectedValue = document.getElementById("selectBox").value;
    var containerRadioButtonsMyModels = document.getElementById("containerRadioButtonsMyModels");
    var containerRadioButtonsPredefinedModels = document.getElementById("containerRadioButtonsPredefinedModels");
    var NoModelsSavedWarning = document.getElementById("NoModelsSavedWarning");

    if (selectedValue == 'none'){
      selectedValue = 'predefinedModels';
    }
    console.log(selectedValue)
    if (selectedValue == 'myModels'){
      containerRadioButtonsMyModels.style.display = "block";
      containerRadioButtonsPredefinedModels.style.display = "none";
    }else{
      containerRadioButtonsMyModels.style.display = "none";
      containerRadioButtonsPredefinedModels.style.display = "block";
    }

  }else if (page == 'trainerMode'){

    var selectedValue = document.getElementById("selectBoxMode").value;
    var containerClassical = document.getElementById("containerClassical");
    var containerQuantum = document.getElementById("containerQuantum");

    if (selectedValue == 'none'){
      selectedValue = 'Classical';
    }

    if (selectedValue == 'Classical'){
      containerClassical.style.display = "block";
      containerQuantum.style.display = "none";
    }else{
      containerClassical.style.display = "none";
      containerQuantum.style.display = "block";
    }

  }else if (page == 'trainerSVmLastLayer'){
    var selectedValue = document.getElementById("selectBoxSVM").value;

    var ocultarCuandoSVM = document.getElementById("ocultarCuandoSVM");

    var selectBoxMode = document.getElementById("selectBoxMode");
    var disabledMode = document.getElementById("disabledMode");

    var selectBoxPretrained = document.getElementById("selectBoxPretrained");
    var disabledPretrained = document.getElementById("disabledPretrained");

    var option1ModelsTL = document.getElementById("option1ModelsTL");
    var option2ModelsTL = document.getElementById("option2ModelsTL");

    var containerSVMchosen = document.getElementById("containerSVMchosen");

    if (selectedValue == 'NO'){
      ocultarCuandoSVM.style.display = "block";

      selectBoxMode.style.display = "block";
      disabledMode.style.display = "none";

      selectBoxPretrained.style.display = "block";
      disabledPretrained.style.display = "none";

      option1ModelsTL.style.display = "block";
      option2ModelsTL.style.display = "none";

      containerSVMchosen.style.display = "none";

      //document.getElementById('cardPretrained').innerHTML = '<h6 align="center">Import the model as pretrained?</h6><p>&ensp;</p><select id="selectBoxPretrained" name="selectBoxPretrained" class="form-select form-select-lg mb-3" aria-label=".form-select-lg example"><option selected value="YES">YES</option><option value="NO">NO</option></select>';
      //document.getElementById('cardMode').innerHTML = '<h6 align="center">Mode</h6><p>&ensp;</p><select id="selectBoxMode" name="selectBoxMode" class="form-select form-select-lg mb-3" aria-label=".form-select-lg example" onchange="getSelectValue('trainerMode')"><option selected value="Classical">Classical</option><option value="Quantum">Quantum</option></select>';
    
    }else{
      ocultarCuandoSVM.style.display = "none";

      selectBoxMode.style.display = "none";
      disabledMode.style.display = "block";

      selectBoxPretrained.style.display = "none";
      disabledPretrained.style.display = "block";

      option1ModelsTL.style.display = "none";
      option2ModelsTL.style.display = "block";

      containerSVMchosen.style.display = "block";

      //document.getElementById('cardPretrained').innerHTML = '<h6 align="center">Import the model as pretrained?</h6><p>&ensp;</p><p style="color: #e06666;"><b>Option disabled when using SVM as last layer.</b></p>';
      //document.getElementById('cardMode').innerHTML = '<h6 align="center">Mode</h6><p>&ensp;</p><p style="color: #e06666;"><b>Option disabled when using SVM as last layer.</b></p>';
    }

  }else if (page == 'typeQuantum'){
    var selectedValue = document.getElementById("selectBoxTypeQuantum").value;
    var linkQuantumKernels = document.getElementById("linkQuantumKernels");
    var linkVariationalQuantumCircuits = document.getElementById("linkVariationalQuantumCircuits");

    if (selectedValue == 'none'){
      selectedValue = 'VGG16 + Quantum Kernel';
    }

    if (selectedValue == 'VGG16 + Quantum Kernel'){
      linkQuantumKernels.style.display = "block";
      linkVariationalQuantumCircuits.style.display = "none";
    }else{
      linkQuantumKernels.style.display = "none";
      linkVariationalQuantumCircuits.style.display = "block";
    }

  }else if (page == 'typeKernel'){
    var selectedValue = document.getElementById("selectKernelSVM").value;
    var disabledGamma = document.getElementById("disabledGamma");
    var selectGammaSVM = document.getElementById("selectGammaSVM");
    
    if (selectedValue == 'linear'){
      disabledGamma.style.display = "block";
      selectGammaSVM.style.display = "none";
    }else{
      disabledGamma.style.display = "none";
      selectGammaSVM.style.display = "block";
    }
  }
}

function getSelectValueDatasetTrainer(){ //se usa para mostrar los radioButtons adecuados dependiendo si esta seleccionado Predefined Models o My Models
  var selectedValue = document.getElementById("selectBoxDataset").value;
  var containerLinkDataset1 = document.getElementById("containerLinkDataset1");
  var containerLinkDataset2 = document.getElementById("containerLinkDataset2");
  var containerLinkDataset3 = document.getElementById("containerLinkDataset3");

  if (selectedValue == 'none'){
    selectedValue = 'Cat and Dog nº1';
  }

  console.log(selectedValue)
  if (selectedValue == 'Cat and Dog nº1'){
    containerLinkDataset1.style.display = "block";
    containerLinkDataset2.style.display = "none";
    containerLinkDataset3.style.display = "none";

  }else if (selectedValue == 'Cat and Dog nº2'){
    containerLinkDataset1.style.display = "none";
    containerLinkDataset2.style.display = "block";
    containerLinkDataset3.style.display = "none";
    
  }else{
    containerLinkDataset1.style.display = "none";
    containerLinkDataset2.style.display = "none";
    containerLinkDataset3.style.display = "block";
  }
}

function showLoadingModelMessage(){ //se usa para mostrar un mnesaje de Loadin model...
  var loadingModelMessage = document.getElementById("loadingModelMessage");

    loadingModelMessage.style.display = "block";
}

function changeHistory(button){ //se usa para mostrar un mnesaje de Loadin model...
  var containerReportsPDF = document.getElementById("containerReportsPDF");
  var containerModelsSaved = document.getElementById("containerModelsSaved");

  if (button == 'PDFreports'){
    containerReportsPDF.style.display = "block";
    containerModelsSaved.style.display = "none";
  }else{
    containerReportsPDF.style.display = "none";
    containerModelsSaved.style.display = "block";
  }
}






