https://github.com/scrosby/fedone/blob/master/whitepapers/operational-transform/operational-transform.rst
원문 : https://en.wikipedia.org/wiki/Operational_transformation

###Operational transformation
OT는 원래 일관성 유지와 평문 문서를 동시 편집시 동시성 제어를 위해 개발되었다. 20여개의 연구가 그 능력을 확장 시켰고, 응용분야를 Group Undo, Locking, 충돌 해결, 명령어 알림 및 압축, 그룹 인식, 트리 구조 문서 편집(XML, HTML), 공동 오피스 생산성 도구 등으로 넓혔다.

Operational transformation
- History
- System architecture
- Basics
- Consistency models (항상성 유지 모델)
 - CC Model
 - The CCI model
 - The CSM model
 - The CA model
 
##History
OT 는 C. Ellis and S. Gibbs[1] in the GROVE 에 의해 처음 개발 되었다. 몇 년후, 정확성 이슈가 제기되었고, 여러가지의 접근방식이 이 이슈를 해결하기 위해 제안되었다. 

##System architecture
 OT 사용한 공동편집은 일반적으로 인터넷과 같은 높은 Latency 환경에서 좋은 민감도를 보장하기 위해 공유 문서의 저장소를 흉내낸 구조를 사용한다.
공유 문서는 각 공동편집 클라이언트의 로컬 저장소에 흉내내어 진다. 그래서 편집 명령어는 즉시 로컬에서 실행 될수 있고, 그 후에 다른곳으로 전파된다.
로컬영역에 도착한 원격지 편집 명령어는 일반적으로 변형되어 진후에 실행된다.
이 명령어 변형은 프로그램이 원하는 동기화가 수준을 모든 클라이언트에서 만족함을 보장한다.
OT는 Lock-Free, non-Blocking 기능은 로컬 반응 시간이 네트워크 Latency 에 대해 민감하지 않게 한다.
그 결과 OT는 특히 공동편집 구현하기에 적합하다.
##Basics
OT의 기본 아이디어는 간단한 글자 편집 예제를 통해 설명할 수 있다. 2개의 클라이언트에 "abc"라는 글자가 있는 문서가 복사되어져 있다.
그리고 아래와 같은 2개의 편집동작이 각각의 클라이언트에서 발생하였다.
 O1 = insert [ 0, "x" ] -> 0번째 자리에 'x'를 입력하라.
 O2 = delete [ 2, 1]  -> 2번째 자리에서 한글자를 지워라.
2개의 명령어는 O1, O2 의 순서로 실행되었다고 가정해 보자.
O1 이 실행된 후에, 문서는 "xabc" 가 된다. O1 실행후, O2 를 실행기 위해서, O2 는 O1 에 대해서 반드시 변형되어야 한다. O2' = Delete[3, 1]가 되어야한다.
O1에 의해 'X' 라는 한글자가 입력되었기 때문에, 위치 인자가 하나 증가하여야 한다.
"xabc"인 문서에서 O2'를 실행하면, 올바른 문자 'c' 를 지우고, 문서는 'xab'가 된다.
그러나 만약 O2 가 변환 없이 실행되면, 'c'가 아니라 'b' 를 지우게 된다. 
OT의 기본적인 원리리는 이전에 실행된 동시 명령어의 영향에 의해 편집 명령어의 인자를 변환 또는 조정하는 것이다. 
변환된 명령어는  문서의 항상성 유지하고, 똑바른 영향을 달성할 수 있다.
 
##Basic idea behind OT
 
#Consistency models (항상성 유지 모델)
OT의  기능 중 하나는 공동편집에서 항상성 유지를 지원하는 것이다. 연구자 커뮤니티에서 여러가지 항상성 모델이 제안되었다. 어떤한 것을 공동편집에서 일반적이고, 어떠한 것들은 OT 알고리즘만을 위한 특별한 것이다.
 
#CC Model
paper "Concurrency control in groupware systems" 논문에서, 공동편집 시스템을 위해 2개의 항상성 요소가 필요하고 하였다,
Precedence(Causality) Property (인과관계) : 공동편집 과정중에  인과관계가 있는 명령어의 실행순서는, 그들의 자연스러운 인과 순서와 같아야한다. 2개의 명령어 간의 인과관계는  Lamport's "happened-before" 관계에서 정의된다.
2개의 명령어가 인과과계적으로 의존성이 없으면, 그들은 동시성을 가진다. 2개의 동시성 명령어는 2개의 다른 문서 복사본에서 다른 순서로 실행될 수 있다.
Convergence(수렴-모든 문서가 하나의 문서로 수렴) : 공유 문서의 흉내내어진 복사본은 정지상태에서 모든 클라이언트에서 동일해야한다. (모든 생성된 명령어가 모든 클라이언트에서 실행된 된 경우)

동시성 명령어가 다른 순서로 실행되지고 나서부터, 편집 명령어들은 교환법칙이 보통 섭립하지 않고, 다른 클라이언트의 문서복사본들은 분기를 탄다.(내용이 달라진다.)
첫 OT 알고리즘은 그룹 문서편집기에서 수렴(convergence)을 달성하기 위해, 논문에서 제안되었다. 우선순위를 지켜주기 위해 state-vector 가 사용되었다.
 
#The CCI model
CCI 모델은 공동편집에서 항상성 관리를 위해 제안되었다. CCI 모델어서는 3개의 항상성 요소가 있다.
Causality preservation(인과관계 보존) : CC 모델에서 Precedence  와 같다.
Convergence: CC 모델에서 Convergence와 같다.
Intention preservation(의도 유지): 어떠한 문서상태에서도 명령어의 수행 효과가 명령어의 의도와 같아야한다. 명령어 O  의 의도는 O가 만들어졌던 문서 상태에서 O가 적용됨으로써 달성될수 있는 실행 결과로 정의된다.
CCI모델은 CC모델을 새로운 기준을 넣어 확장한 것이다. Convergence와 Intention preservation의 중요한 차이점은 Convergence은 항상 시리얼라이즈 프로토콜에 의해 달성할수 있다라는 점이다. 하지만  Intention preservation 는 명령어가 원래의 형태에서 항상 실행된다면, 어떤 시리얼라이즈 프로토콜에 의해서는 달성될수 없을 수도 있다. non 시리얼라이드 intention preservation 요소를 달성하는 것은 중요한 기술적 도전이다. OT는 공동편집에서 intention preservation와 Convergence을 달성하는데 적합하다. 
CCI모델은 문서타입, 데이터 모델, 명령어 타입, 지원기술(OT, 시리얼라이즈, Undo,Redo ) 등에 독립적이다. 특정 데이터, 특정 프로그램을 위해 설계된 기술을 위한 정확성 검증을 위해 만들어진것이 아니다. 
intention preservation 의 개념은 아래 3가지 레벨에서 정의 된다.
첫째, 공동편집을 위해 일반적인 항상성 요구사항으로 정의 된다.
둘째, 일반적인 OT  동작을 위한 전후 변환 조건으로 정의 된다.
셋째, 2개의 원본 명령어를 위한 OT 동작을 위한 설계를 가이드 하기 위한 특정 명령어 검증 기준으로 정의된다.

#The CSM model
intention preservation 의 조건은 공식적인 증거의 목적으로 CCI 모델에서 공식으로 특별한것이 아니다.
SDT,LBT 접근 법은 주어질수 있는 다른 조건을 형식화하도룩 시도했다. 이 2가지의 접근법에서 항상성 모델은 아래와 같은 일반화된 조건으로 구성되어 있다.
 
Causality: CC 모델에서 Precedence  와 같다.
Single-operation effects: 어떠한 실행 클라이언트에서 어떠한 실행결과가 만든 클라이언트와 같은 효과를 달성해야한다.
Multi-operation effects: t어떠한 2개의 명령어의 실행결과 관계가 어떠한 상태에서 2개다 실행된후에 유지된다.

#The CA model
위의 CSM 모델은 시스템 안에서 모든 오브제극의 전체 순서가 특별해야 함이 필요하다. 효과적으로, 이 사양은 삽입 동작에 의해 도입된 새로운 오브젝트를 줄인다.
그러나 전체 순서에 사양은 삽입 관계를 끈는 거 같은 프로그램 종속적인 정책을 수반한다. (같은 위치에 2개의 명령어의 의해 삽입된 새로운 객체)
그 결과, 전체 순서는 프로그램 종속적이다. 더욱이 알고리즘에서 전체 순서는 변환 함수에서 유지 되어야 한다. (이동작은 알고리즘의 시작복잡도를 높힌다.)
그 대신에 CA 모델은 admissibility theory["Commutativity-Based Concurrency Control in Groupware". Proceedings of the First IEEE Conference on Collaborative Computing: Networking, Applications and Worksharing (CollaborateCom'05).] 에 근거를 둔다.
CA 모델은 2가지 방향을 가진다.
Causality: CC 모델에서 Precedence  와 같다.
Admissibility(인정되는 ): 모든 명령어의 발동은 그것의 동작 환경에서 허용되어야한다. 예를 들어) 더 빠른 발동의 의해 인정된 모든 어떠한 효과관계를 침범하지 않아야한다. (오브젝트 오더링 같은...)

These two conditions imply convergence. All cooperating sites converge in a state in which there is a same set of objects that are in the same order. Moreover, the ordering is effectively determined by the effects of the operations when they are generated. Since the two conditions also impose additional constraints on object ordering, they are actually stronger than convergence. The CA model and the design/prove approach are elaborated in the 2005 paper.[9] It no longer requires that a total order of objects be specified in the consistency model and maintained in the algorithm, which hence results in reduced time/space complexities in the algorithm.
 
