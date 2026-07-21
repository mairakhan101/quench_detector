
mfun_saveClear

pointsZi(1,1)=0.0+0.075*i;
pointsZi(2,1)=0.0+0.076*i;



measRadius=0.048383

%% 6p_normal winding
orientAngle=0*pi/180;
%% get coilparameters for QA
coilparamQA='coilparam_MQXFA_test_6p_norm.dat'
QA=importdata(coilparamQA);
QA.data(:,4)=QA.data(:,4)-orientAngle;
%% writeCoilparam('coilparam_MQXFA_6p_norm.dat',QA)
plotCoilparamVert(QA.data,coilparamQA)
QAvertZ=QA.data(:,3).*exp(i*QA.data(:,4))
%% get flux using the coilparam data and the current lines which cause the field
  Avp_6pN=fluxCalc(QAvertZ,pointsZi);
%% set some flux contributions to be positive and some negative (the wires in the coilparam alternate in sign as they are understood to be a loop) 
alt_pm1=reshape([ones(1,length(Avp_6pN)/2)',ones(1,length(Avp_6pN)/2)'*-1]',1,length(Avp_6pN))';
%%calculate the total normal6p flux by summing the contributions from all the wires
norm6p=sum(Avp_6pN.*alt_pm1)
%% write and plot the flux
dlmwrite(['flux_',coilparamQA],[QA.data(:,4),real(Avp_6pN.*alt_pm1)],'delimiter',' ','precision','%.10e')
figure;plot(real(Avp_6pN),'-bo')
title(['Flux vs Angle Number 6p norm','(arbitrary units)'])
%mfun_fluxDataFFT(size(Avp,1),real(Avp),30)


%% do the same for the 6p_skew winding
orientAngle=0*pi/180;
coilparamQA='coilparam_MQXFA_test_6p_skew.dat'
QA=importdata(coilparamQA);
QA.data(:,4)=QA.data(:,4)-orientAngle;
%% writeCoilparam('coilparam_MQXFA_6p_skew.dat',QA)
plotCoilparamVert(QA.data,coilparamQA)
QAvertZ=QA.data(:,3).*exp(i*QA.data(:,4))
  Avp_6pS=fluxCalc(QAvertZ,pointsZi);
alt_pm1=reshape([ones(1,length(Avp_6pS)/2)',ones(1,length(Avp_6pS)/2)'*-1]',1,length(Avp_6pS))';
skew6p=sum(Avp_6pS.*alt_pm1)
dlmwrite(['flux_',coilparamQA],[QA.data(:,4),real(Avp_6pS.*alt_pm1)],'delimiter',' ','precision','%.10e')
figure;plot(real(Avp_6pS),'-bo')
title(['Flux vs Angle Number 6p skew','(arbitrary units)'])
%mfun_fluxDataFFT(size(Avp,1),real(Avp),30)



%% 8p_normal winding
orientAngle=0*pi/180;
coilparamQA='coilparam_MQXFA_test_8p_norm.dat'
QA=importdata(coilparamQA);
QA.data(:,4)=QA.data(:,4)-orientAngle;
%% writeCoilparam('coilparam_MQXFA_8p_norm.dat',QA)
plotCoilparamVert(QA.data,coilparamQA)
QAvertZ=QA.data(:,3).*exp(i*QA.data(:,4))
  Avp_8pN=fluxCalc(QAvertZ,pointsZi);
alt_pm1=reshape([ones(1,length(Avp_8pN)/2)',ones(1,length(Avp_8pN)/2)'*-1]',1,length(Avp_8pN))';
norm8p=sum(Avp_8pN.*alt_pm1)
dlmwrite(['flux_',coilparamQA],[QA.data(:,4),real(Avp_8pN.*alt_pm1)],'delimiter',' ','precision','%.10e')
figure;plot(real(Avp_8pN),'-bo')
title(['Flux vs Angle Number 8p norm','(arbitrary units)'])
%mfun_fluxDataFFT(size(Avp,1),real(Avp),30)

%% 8p_skew winding
orientAngle=0*pi/180;
coilparamQA='coilparam_MQXFA_test_8p_skew.dat'
QA=importdata(coilparamQA);
QA.data(:,4)=QA.data(:,4)-orientAngle;
%% writeCoilparam('coilparam_MQXFA_8p_skew.dat',QA)
plotCoilparamVert(QA.data,coilparamQA)
QAvertZ=QA.data(:,3).*exp(i*QA.data(:,4))
  Avp_8pS=fluxCalc(QAvertZ,pointsZi);
alt_pm1=reshape([ones(1,length(Avp_8pS)/2)',ones(1,length(Avp_8pS)/2)'*-1]',1,length(Avp_8pS))';
skew8p=sum(Avp_8pS.*alt_pm1)
dlmwrite(['flux_',coilparamQA],[QA.data(:,4),real(Avp_8pS.*alt_pm1)],'delimiter',' ','precision','%.10e')
figure;plot(real(Avp_8pS),'-bo')
title(['Flux vs Angle Number 8p skew','(arbitrary units)'])
%mfun_fluxDataFFT(size(Avp,1),real(Avp),30)




C6p=real(norm6p)+i*real(skew6p)
C8p=real(norm8p)+i*real(skew8p)

%% technically should find the FIELDS by using the sensitvity,
%% but since the sensitivity is calculated on the same radius for all windings, 
%% if have single current line can just take out the ratio of 8p to 6p and then can just calculate directly from the harmonics
%% qloc=C6p/C8p*measRadius


qloc=4/3*C6p/C8p*measRadius






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% calculate flux using all QA vertices with the vector potential from all the current wires
function Avp = fluxCalc(QAvertZ,pointsZi)
    for jjvert=1:length(QAvertZ)
	zEval(jjvert,1)=QAvertZ(jjvert);
        Avp(jjvert,1)=0;

	for jji=1:length(pointsZi)
	  Avp(jjvert,1)=Avp(jjvert,1)+(-1)^jji*-1*log(zEval(jjvert)-pointsZi(jji));
	end

    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotCoilparamVert(coilparamData,titleText)

    qq=coilparamData;

    npts=length(qq)
    startPt=1
    endPt=2
    figure
    for jjplot=1:2:length(qq)
      hold on;plot(qq(jjplot:jjplot+1,3).*cos(qq(jjplot:jjplot+1,4)),qq(jjplot:jjplot+1,3).*sin(qq(jjplot:jjplot+1,4)),'-o')
    end

    for jjplot=1:2:length(qq)
      hold on;plot(qq(jjplot,3).*cos(qq(jjplot,4)),qq(jjplot,3).*sin(qq(jjplot,4)),'rs','MarkerFaceColor','r')
    end

    for jjplot=2:2:length(qq)
      hold on;plot(qq(jjplot,3).*cos(qq(jjplot,4)),qq(jjplot,3).*sin(qq(jjplot,4)),'ks','MarkerFaceColor','k')
    end

    title(strrep(titleText,'_','\_'))

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function writeCoilparam(coilparamName,coilparamStruct)
    outFile=fopen(coilparamName,'w'); 
    fprintf(outFile,'%s \n',coilparamStruct.textdata{1});
    for jp=1:size(coilparamStruct.data,1)
       fprintf(outFile,'%s  %15.10e  %10.2f  %15.10e  %15.10e \n','VERTEX RBUCK',coilparamStruct.data(jp,1),coilparamStruct.data(jp,2),coilparamStruct.data(jp,3),coilparamStruct.data(jp,4));
    end
    fclose(outFile);
end
