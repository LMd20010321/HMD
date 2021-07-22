//EK算法
#include<bits/stdc++.h>

using namespace std;

const int N=1010,M=20010,INF=1e8;

int n,m ,S,T;
int h[N],e[M],f[M],ne[M],idx;
int d[N],pre[N];
bool st[N];

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}

bool bfs()
{
    queue<int> q;
    memset(st,false,sizeof st);
    q.push(S),st[S]=true,d[S]=INF;

    while(!q.empty())
    {
        int t=q.front();
        for(int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if(!st[ver]&&f[i])
            {
                st[ver]=true;
                d[ver]=min(d[t],f[i]);
                pre[ver]=i;//前驱边
                if(ver==T) return true;
                q.push(ver);
            }
        }
        q.pop();
    }
    return false;
}

int EK()
{
    int r=0;
    while(bfs())
    {
        r+=d[T];
        for(int i=T;i!=S;i=e[pre[i]^1])
        {
            f[pre[i]]-=d[T],f[pre[i]^1]+=d[T];
        }
    }
    return r;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie();cout.tie();

    cin>>n>>m>>S>>T;
    memset(h,-1,sizeof h);
    while(m--)
    {
        int a,b,c;
        cin>>a>>b>>c;
        add(a,b,c);
    }

    cout<<EK()<<endl;
    return 0;
}
/*
 最大流最小割定理
（1）流f是最大流
（2）残留网络中不存在增广路径
（3）存在某一个割（S,T)，使得 |f|=c（S,T)
 */
