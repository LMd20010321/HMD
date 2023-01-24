using TSSOS
using DynamicPolynomials

n=7
@polyvar x[1:2*n-6]

f=1
for i=1:n-4
    global f+=0.5*(x[i]*x[n-3+i+1]-x[i+1]*x[n-3+i])
end

pop=[f]
for i=1:2*n-6
    push!(pop,x[i]-1)
end

r=2
m=2*3.1415926535/n
theta=0
px=[]
py=[]
for i=1:n
    push!(px,(1-cos(theta))*r)
    push!(py,(1-sin(theta))*r)
    global theta+=m
end

vx=Any[0,0,1]
vy=Any[1,0,0]
for i=1:n-3
    push!(vx,x[i])
    push!(vy,x[n-3+i])
end

for i=1:n
    for j=i+1:n
        for k=j+1:n
            ans1 = px[i]*py[j]+px[k]*py[i]+px[j]*py[k]-px[k]*py[j]-px[j]*py[i]-px[i]*py[k]
            ans2 = vx[i]*vy[j]+vx[k]*vy[i]+vx[j]*vy[k]-vx[k]*vy[j]-vx[j]*vy[i]-vx[i]*vy[k]
            push!(pop,ans1*ans2)
        end
    end
end

ax=Any[0,1]
ay=Any[0,0]
for i=1:n-3
    push!(ax,x[i])
    push!(ay,x[n-3+i])
end
ax=[ax;[0,0]]
ay=[ay;[1,0]]

for i=1:n-1
    push!(pop,ax[i]*ay[i+1]-ax[i]*ay[i+2]-ax[i+1]*ay[i]+ax[i+2]*ay[i]+ax[i+1]*ay[i+2]-ax[i+2]*ay[i+1]-1)
end

d=3
cnt=n-1
opt,sol,data=cs_tssos_first(pop,x,d,numeq=cnt,TS="MD",solver="Mosek")
opt,sol,data = cs_tssos_higher!(data, TS="MD")
print(data.flag)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
