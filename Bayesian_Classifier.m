%% stage 1
x_w1=[ -3.9847 -3.5549 -1.2401 -0.9780 -0.7932 -2.8531 -2.7605 -3.7287 -3.5414  ...
    -2.2692 -3.4549 -3.0752 -3.9934  -0.9780 -1.5799 -1.4885 -0.7431 -0.4221 ...
    -1.1186 -2.3462 -1.0826 -3.4196 -1.3193 -0.8367 -0.6579 -2.968];%w1条件下的类条件概率密度的采样结果
x_w2=[2.8792 0.7932 1.1882 3.0682 4.2532 0.3271,0.9846 2.7648 2.6588];%w2 条件下的类条件概率密度的采样结果
pw1=0.9;
pw2=0.1;
u1=mean(x_w1);
u2=mean(x_w2);
a1=std(x_w1);
a2=std(x_w2);
syms x;
pxw1=gaussmf(x,[a1 u1]);%求观察到的特征在w1条件下的概率
pxw2=gaussmf(x,[a2 u2]);%求观察到的特征在w2条件下的概率
pw1x=(pw1*pxw1)/(pw1*pxw1+pw2*pxw2);
pw2x=(pw2*pxw2)/(pw1*pxw1+pw2*pxw2);
% d=double(solve(pw1x-pw2x));%不考虑风险
% x=-5:0.1:5;
% plot(x,subs(pw1x));
% hold on
% plot(x,subs(pw2x));
% x=d(2);
% y=subs(pw2x);
% plot([x,x],[0,y],'r--')
% text(0.4,0.03,'0.7239')
% legend('P(w1|x)','P(w2|x)');
% xlabel('x');
% ylabel('p');
% title('the posterior probability density functions')

risk=[0 1;6 0];
pmat=[pw1x;pw2x];
r=risk*pmat;
d=double(solve(r(1)-r(2)));
x=-5:0.1:5;
plot(x,subs(r(1)));
hold on
plot(x,subs(r(2)));
x=d(2);
y=subs(r(1));
plot([x,x],[0,y],'r--')
text(1.2,0.2,'1.3738')
legend('R(a1|X)','R(a2|X)');
xlabel('x');
ylabel('r');
title('the decision risk probability density functions')

%% deep study
pwr1=1/3;
pwr2=1/3;
pwr3=1/3;

mu1=[0,5];%数学期望
sigma1=[1 0;0,1];%协方差矩阵
r1=mvnrnd(mu1,sigma1,200);

mu2=[-3,0];%数学期望
sigma2=[1 0;0,1];%协方差矩阵
r2=mvnrnd(mu2,sigma2,200);

mu3=[3,0];%数学期望
sigma3=[1 0;0,1];%协方差矩阵
r3=mvnrnd(mu3,sigma3,200);
% plot(r1(:,1),r1(:,2),'r*')
% hold on
% plot(r2(:,1),r2(:,2),'g*')
% plot(r3(:,1),r3(:,2),'y*')
% title('The Distribution of Our Data')
% xlabel('x');
% ylabel('y');

syms x1 x2 z;
x=[x1 x2];
pxwr1=mvnpdf(x,mu1,sigma1);%求观察到的特征在w1条件下的概率
pxwr2=mvnpdf(x,mu2,sigma2);
pxwr3=mvnpdf(x,mu3,sigma3);

pw1xr=(pwr1*pxwr1)/(pwr1*pxwr1+pwr2*pxwr2+pwr3*pxwr3);
pw2xr=(pwr2*pxwr2)/(pwr1*pxwr1+pwr2*pxwr2+pwr3*pxwr3);
pw3xr=(pwr3*pxwr3)/(pwr1*pxwr1+pwr2*pxwr2+pwr3*pxwr3);
d12=solve(pw1xr-pw2xr,x2);%不考虑风险
d13=solve(pw1xr-pw3xr,x2);
d32=solve(pw3xr-pw2xr,x2);

plot(r1(:,1),r1(:,2),'r*')
hold on
plot(r2(:,1),r2(:,2),'g*')
plot(r3(:,1),r3(:,2),'y*')
x1=-5:0.1:5;
d12=eval(d12);
d13=eval(d13);
k=find((d12==d13)==1);

plot([-5,x1(k)],[d12(1),d12(k)],'r--');
plot([x1(k),5],[d13(k),d13(end)],'r--');
plot([0,0],[-5,d12(k)],'r--'); %d32为空，说明x2存在多个取值
xlabel('x1');
ylabel('x2');
title('The Decision Boundary of Our Data ')


% plot3(x1,x2,subs(pw1xr));
% % hold x()on
% % plot3(x1,x2,subs(pw2xr));
% 
% % plot([x,x],[0,y],'r--')
% % text(0.4,0.03,'0.7239')
% % legend('P(w1|x)','P(w2|x)');
% % xlabel('x');
% % ylabel('p');
% % title('the posterior probability density functions')
