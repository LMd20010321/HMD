//dinic
#include<bits/stdc++.h>

using namespace std;

const int N=10010,M=200010,INF=1e8;

int n,m,S,T;
int h[N],e[M],f[M],ne[M],idx;
int d[N],cur[N];//cur为当前弧优化

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}

bool bfs()
{
    queue<int> q;
    memset(d,-1,sizeof d);
    q.push(S);
    d[S]=0;
    cur[S]=h[S];
    while(!q.empty())
    {
        int t=q.front();
        for(int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if(d[ver]==-1&&f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];//当前弧初始化
                if(ver==T)
                    return true;
                q.push(ver);
            }
        }
        q.pop();
    }
    return false;
}

int dfs(int u,int limit)
{
    if(u==T) return limit;
    int flow=0;
    for(int i=cur[u];~i&&flow<limit;i=ne[i])
    {
        cur[u]=i;//更新当前弧
        int ver=e[i];
        if(d[ver]==d[u]+1&&f[i])
        {
            int t=dfs(ver,min(f[i],limit-flow));
            if(!t) d[ver]=-1;//当前边不可用，删掉这个点
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}

int dinic()
{
    int r=0,flow;
    while(bfs())//找到增广路径时，更新r
    {
        while(flow=dfs(S,INF))
            r+=flow;
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

    cout<<dinic()<<endl;
    return 0;
}
