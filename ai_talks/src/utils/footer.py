from pathlib import Path

import streamlit as st

from .constants import BUG_REPORT_URL, REPO_URL
from .helpers import render_svg

#st.text("Командой ЦАР был разработан прототип ЦАР-ассистента - помощника на основе искусственного интеллекта. 
        #ЦАР-ассистент успешно справляется с задачами, связанными с генерацией программного кода, код-ревью и в настоящее
        #время обучается, как старательный студент, чтобы изучить банковские ЛПА и в дальнейшем стать экспертом по ЛПА.
        #Мы уверены, что ЦАР-ассистент станет надежным помощником для решения рутинных задач в банке.")

        
def show_info(icon: Path) -> None:
    st.divider()
    #st.markdown(f"<div style='text-align: justify;'>"ЦАР-ассистента - помощника"</div>",
                #unsafe_allow_html=True)
    #st.text("Командой ЦАР был разработан прототип ЦАР-ассистента - помощника на основе искусственного интеллекта. 
        #ЦАР-ассистент успешно справляется с задачами, связанными с генерацией программного кода, код-ревью и в настоящее
        #время обучается, как старательный студент, чтобы изучить банковские ЛПА и в дальнейшем стать экспертом по ЛПА.
        #Мы уверены, что ЦАР-ассистент станет надежным помощником для решения рутинных задач в банке.")
        
    #st.markdown(f"<div style='text-align: justify;'>{st.session_state.locale.responsibility_denial}</div>",
                #unsafe_allow_html=True)
    #st.divider()
    #st.markdown(f"""
         #:page_with_curl: {st.session_state.locale.footer_title}
       
        #- {render_svg(icon)} [{st.session_state.locale.footer_channel}](https://https://t.me/Br_Evgeny)
       
    #""", unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify;">Командой ЦАР был разработан прототип ЦАР-ассистента - помощника на основе искусственного интеллекта. ЦАР-ассистент успешно справляется с задачами, связанными с генерацией программного кода, код-ревью и в настоящее время обучается, как старательный студент, чтобы изучить банковские ЛПА и в дальнейшем стать экспертом по ЛПА. Мы уверены, что ЦАР-ассистент станет надежным помощником для решения рутинных задач в банке.</div>', unsafe_allow_html=True)
    #st.markdown(f"<div style='text-align: justify;'>{st.session_state.locale.responsibility_denial}</div>", unsafe_allow_html=True)
    st.divider()
    #st.markdown(f"project [repo on github]({REPO_URL}) waiting for your :star: | [report]({BUG_REPORT_URL}) a bug")


def show_donates() -> None:
    #st.markdown(f"""
        ### :moneybag: {st.session_state.locale.donates}

        #**Crypto:**
        #- Bitcoin (BTC)
        #```
        #1HRDUif7oKDw9XJFXZ14TZZazokf4QH9fb
        #```
        #- USD Tether (USDT TRC20):
        #```
        #TMQ5RiyQ7bv3XjB6Wf6JbPHVrGkhBKtmfA
        #```
        #- Toncoin (TON):
        #```
        #UQDbnx17N2iOmxfQF0k55QScDMB0MHL9rsq-iGB93RMqDhIH
        #```

        #**{st.session_state.locale.donates2}:**
        #- [Buy Me A Coffee](https://www.buymeacoffee.com/aitalks)
        #- [ko-fi](https://ko-fi.com/ai_talks)
        #- [PayPal](https://www.paypal.com/paypalme/aitalks)
    #""")
    #st.markdown(f"""
        #**{st.session_state.locale.donates1}:**
        #- [Tinkoff](https://www.tinkoff.ru/cf/4Ugsr5kQ1sR)
        #- [donationalerts](https://www.donationalerts.com/r/if_ai)
        #- [boosty](https://boosty.to/ai-talks/donate)
        #- [CloudTips](https://pay.cloudtips.ru/p/eafa15b2)
    #""")
    #_, img_col, _ = st.columns(3)
    with img_col:
        st.image("ai_talks/assets/qr/tink.png", width=200)
    #st.divider()
    st.markdown(f"<div style='text-align: justify;'>{st.session_state.locale.donates_info}</div>",
                unsafe_allow_html=True)
