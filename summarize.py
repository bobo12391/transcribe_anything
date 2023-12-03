# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 12:31:29 2023

@author: Administrator
"""
import dashscope
from dashscope import Generation
from transformers import AutoTokenizer, AutoModel

class GLM():
    def __init__(self, model_path: str="glm/chatglm3-6b"):
        self.model_path = model_path
                        
    def get_translation_answer(self, question: str) -> str:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).quantize(4).cuda()
        self.model = self.model.eval()
        self.prompt = "<|system|>\n\
                        你是一个音频内容总结机器人，\
                        你一段视频时间段和不同人的人物对话，你根据不同发言人，\
                        时间先后顺序以及发言内容做一个概要提炼，\
                        要的是总结不是每句转述，后期将会处理几百回对话的情况所以我们希望能够您能自己学会总结\
                        (每段QA的时间段都不一样)\n"
        type_prompt = "<|user|>\n\
                        SPEAKER_01: 好,没问题。好,接下来我大概一共是两百个问题。\n\
                        SPEAKER_00: 那咱们进行的时候,有时候我要发会儿呆,或者回答的慢,可能是在默念那句话,你要理解。\n\
                        SPEAKER_01: 对,好,第一个问题,在接下来的这个漫长的采访当中,有没有什么问题我不该问,或者我问了,你也不想好好回答?\n\
                        SPEAKER_00: 这些设计公司机密的事不方便讲,我会明说。比如说我说出来可能会伤害别人,这种事情可能就没有办法说。像之前接受媒体采访的时候也有一个问题就是说,他会问到一个问题,他觉得我在回避什么,其实很多时候我们不讲一部分话是为了避免伤害别人,这里边有合作伙伴,有已经离职的同事等等这些情况,所以如果是这两种情况就会不方便讲,其他的我都会尽量百分之百配合。\n\
                        SPEAKER_01: 我和罗永浩有一次在一个咖啡馆聊天,因为那天实在订不到包厢,然后我们俩就坐在大厅里。我们俩心中多少有点感觉,因为是个名人嘛,要不然为啥要订包厢呢?但是那天我记得是圣诞节的下午,那个茶馆里到处都是人,但是从头到尾没有人认识我们俩。\n\
                        SPEAKER_00: 其实还好,我注意到左后方有一个人一直在看你,但他可能比较礼貌没过来给你打招呼,我觉得他还是认出你了。\n\
                        SPEAKER_01: 所以现在因为这个采访也在电视上播出嘛,所以我们不能假设每一个观众都知道罗永浩是谁。那当然。所以接下来我的第二个问题就有点像保安问的,你是谁?你从哪里来?你要向哪里去?\n\
                        SPEAKER_00: 我叫罗永浩。然后我从吉林省延吉市来,然后来到北京快十多年了吧,然后要往科技领袖那个位置上去,大概是这样。\n\
                        SPEAKER_01: 好,那第三个问题就出来了,就在2017年的3月份,你的下一代锤子手机快要发布的时候,你现在给你的公司打分,如果一百分,你会打多少分?\n\
                        SPEAKER_00: 我们到五月十五号就正好五年了,这五年综合的话应该打一个,到今天这个时间点呢,打个七十多分吧,也就是这个样子。因为比起从事这个行业的人出来创业来讲,我们前期学习的时间和周期这个是肯定是比较长的,因为跨界跨得比较不正常。所以走的还是比较慢吧。然后我们这个企业跟别家可能有一个比较大的区别就是很多人公认我们长板是特别长的,比五百强还长,短板呢特别短,比深圳的山寨公司还要短。所以这样综合下来走到今天能平均达到一个七十多分,我自己还是挺满意的,但是离一个真正的优秀企业还是差得很远。对。\n\
                        SPEAKER_01: 那二十多分是怎么扣的?\n\
                        SPEAKER_00: 这些短板没有完全补齐。长板呢,我们原有的优势要继续发挥,但是在那儿,我中间犯过一个比较长的时期的错,就是说在已经长的长板上投入了跟那些短板差不多一样的精力,这是不对的。因为要把那个短板迅速地补齐,长板的优势才能发挥,否则它始终是会被短板给拉下来。\n\
                        SPEAKER_01: 我从最早听你网上流传的那一段MP3,一直到今天咱俩同框对话,有几个词,反正是你的标志性的词,也是你用一己之力把一个相对陌生的词突然变成了一个大众热词,比如说彪悍,我的奋斗,工匠精神,情怀等等。那这些词跟你是如影随形的,那今天这些词哪些你还认,哪些你已经不认了?\n\
                        SPEAKER_00: 其实有的时候就是这样,就是你自己知道的你和真相和公众眼里的你这三件事永远都是三件事。它不是一件事,所以比如说这些年人说贴在我身上标签的那些东西其实跟我没有特别大的关系,比如说彪悍这件事是在新东方偶然一个课上偶尔讲的一句话,那个传出来录音以后是传得最广的,但是对我来讲,我在教书的那几年试图给学生传递的价值观和我的生活态度里边,其实那个是微不足道的一块,但是呢它会被过分强调以至于早年的时候见到任何记者都会问这个东西所以有一阵我还对这个很反感后来后来我就认了因为就我刚才讲的就是真相和你自己眼里的你和别人眼里你这三件事永远永远是三件事所以从这一上很多东西都是我后来也就认了没有办法就认了但是其实跟我没有什么特别大的关系刚才你还提到那个情怀情怀也是我们发第一代的操作系统那个OS那个发布会的时候在那个发布会上全程我就提了这么一句,结果这一句呢被很多偏文青的这种媒体的记者当成一个事儿然后去讲,讲的话呢那些偏理科生呢就觉得很烦,你做科技企业的讲什么情怀,然后拿那个就使劲恶心我,包括我们手机上市以后他们说你这个手机是白送的,情怀就值三千块。\n\
                         <|assistant|>\n\
                         在这段对话中，SPEAKER_00分享了他的一些经历和感悟。首先，他表示自己会阅读各种类型的书籍，包括文学作品、社会科学、历史典籍等。接着，他谈论了自己的创业经历，表示虽然他的公司在创业初期学习时间较长，但仍然取得了中等水平的成绩，综合评分约为70多分。\n\n\
                        然后，SPEAKER_00谈到了他的公司锤子手机在2017年3月即将发布时，被一些人质疑他的手机定价过高。针对这个问题，罗永浩解释说，他的公司在创业初期犯过错误，未能充分发挥长板优势，而把同样级别的精力投入到了短板上。他认为，应该迅速弥补短板，发挥长板优势，否则会被短板拖累。\n\n\
                        此外，SPEAKER_00还谈到了一些与他个人品牌密切相关的词语，如“彪悍”、“奋斗”、“工匠精神”和“情怀”等。他表示，这些词语是用自己的经历和价值观传递给学生的，但并非自己特别在意。\n\n\
                        最后，SPEAKER_00谈到了一些媒体对他的评价和标签，例如“贴吧诗人”和“情怀”等。他表示，这些标签并非完全符合自己，但自己会接受这些评价，因为这些词语反映了他过去的一些观点和价值观。\n\n"
        question = self.prompt + type_prompt + "<|user|>\n" + question + "<|assistant|>\n"
        response, _ = self.model.chat(self.tokenizer, question, history=[])
        return response

class QUWEN():
    def __init__(self, key: str, prompt: str=None) -> None:
        self.key = key
        if prompt:
            self.prompt = prompt
        else:
            self.prompt = "你是一个音频内容总结机器人，\
                            你一段视频时间段和不同人的人物对话，你根据不同发言人，\
                            时间先后顺序以及发言内容做一个概要提炼，\
                            要的是总结不是每句转述，后期将会处理几百回对话的情况所以我们希望能够您能自己学会总结\
                            (每段QA的时间段都不一样)\n"
        self.messages = {'role': 'system', 'content': self.prompt}
        
    def get_translation_answer(self, question: str) -> str:
        try:
            type_prompt = "<|user|>\n\
                        SPEAKER_01: 好,没问题。好,接下来我大概一共是两百个问题。\n\
                        SPEAKER_00: 那咱们进行的时候,有时候我要发会儿呆,或者回答的慢,可能是在默念那句话,你要理解。\n\
                        SPEAKER_01: 对,好,第一个问题,在接下来的这个漫长的采访当中,有没有什么问题我不该问,或者我问了,你也不想好好回答?\n\
                        SPEAKER_00: 这些设计公司机密的事不方便讲,我会明说。比如说我说出来可能会伤害别人,这种事情可能就没有办法说。像之前接受媒体采访的时候也有一个问题就是说,他会问到一个问题,他觉得我在回避什么,其实很多时候我们不讲一部分话是为了避免伤害别人,这里边有合作伙伴,有已经离职的同事等等这些情况,所以如果是这两种情况就会不方便讲,其他的我都会尽量百分之百配合。\n\
                        SPEAKER_01: 我和罗永浩有一次在一个咖啡馆聊天,因为那天实在订不到包厢,然后我们俩就坐在大厅里。我们俩心中多少有点感觉,因为是个名人嘛,要不然为啥要订包厢呢?但是那天我记得是圣诞节的下午,那个茶馆里到处都是人,但是从头到尾没有人认识我们俩。\n\
                        SPEAKER_00: 其实还好,我注意到左后方有一个人一直在看你,但他可能比较礼貌没过来给你打招呼,我觉得他还是认出你了。\n\
                        SPEAKER_01: 所以现在因为这个采访也在电视上播出嘛,所以我们不能假设每一个观众都知道罗永浩是谁。那当然。所以接下来我的第二个问题就有点像保安问的,你是谁?你从哪里来?你要向哪里去?\n\
                        SPEAKER_00: 我叫罗永浩。然后我从吉林省延吉市来,然后来到北京快十多年了吧,然后要往科技领袖那个位置上去,大概是这样。\n\
                        SPEAKER_01: 好,那第三个问题就出来了,就在2017年的3月份,你的下一代锤子手机快要发布的时候,你现在给你的公司打分,如果一百分,你会打多少分?\n\
                        SPEAKER_00: 我们到五月十五号就正好五年了,这五年综合的话应该打一个,到今天这个时间点呢,打个七十多分吧,也就是这个样子。因为比起从事这个行业的人出来创业来讲,我们前期学习的时间和周期这个是肯定是比较长的,因为跨界跨得比较不正常。所以走的还是比较慢吧。然后我们这个企业跟别家可能有一个比较大的区别就是很多人公认我们长板是特别长的,比五百强还长,短板呢特别短,比深圳的山寨公司还要短。所以这样综合下来走到今天能平均达到一个七十多分,我自己还是挺满意的,但是离一个真正的优秀企业还是差得很远。对。\n\
                        SPEAKER_01: 那二十多分是怎么扣的?\n\
                        SPEAKER_00: 这些短板没有完全补齐。长板呢,我们原有的优势要继续发挥,但是在那儿,我中间犯过一个比较长的时期的错,就是说在已经长的长板上投入了跟那些短板差不多一样的精力,这是不对的。因为要把那个短板迅速地补齐,长板的优势才能发挥,否则它始终是会被短板给拉下来。\n\
                        SPEAKER_01: 我从最早听你网上流传的那一段MP3,一直到今天咱俩同框对话,有几个词,反正是你的标志性的词,也是你用一己之力把一个相对陌生的词突然变成了一个大众热词,比如说彪悍,我的奋斗,工匠精神,情怀等等。那这些词跟你是如影随形的,那今天这些词哪些你还认,哪些你已经不认了?\n\
                        SPEAKER_00: 其实有的时候就是这样,就是你自己知道的你和真相和公众眼里的你这三件事永远都是三件事。它不是一件事,所以比如说这些年人说贴在我身上标签的那些东西其实跟我没有特别大的关系,比如说彪悍这件事是在新东方偶然一个课上偶尔讲的一句话,那个传出来录音以后是传得最广的,但是对我来讲,我在教书的那几年试图给学生传递的价值观和我的生活态度里边,其实那个是微不足道的一块,但是呢它会被过分强调以至于早年的时候见到任何记者都会问这个东西所以有一阵我还对这个很反感后来后来我就认了因为就我刚才讲的就是真相和你自己眼里的你和别人眼里你这三件事永远永远是三件事所以从这一上很多东西都是我后来也就认了没有办法就认了但是其实跟我没有什么特别大的关系刚才你还提到那个情怀情怀也是我们发第一代的操作系统那个OS那个发布会的时候在那个发布会上全程我就提了这么一句,结果这一句呢被很多偏文青的这种媒体的记者当成一个事儿然后去讲,讲的话呢那些偏理科生呢就觉得很烦,你做科技企业的讲什么情怀,然后拿那个就使劲恶心我,包括我们手机上市以后他们说你这个手机是白送的,情怀就值三千块。\n\
                        <|assistant|>\n\
                         在这段对话中，SPEAKER_00分享了他的一些经历和感悟。首先，他表示自己会阅读各种类型的书籍，包括文学作品、社会科学、历史典籍等。接着，他谈论了自己的创业经历，表示虽然他的公司在创业初期学习时间较长，但仍然取得了中等水平的成绩，综合评分约为70多分。\n\n\
                        然后，SPEAKER_00谈到了他的公司锤子手机在2017年3月即将发布时，被一些人质疑他的手机定价过高。针对这个问题，罗永浩解释说，他的公司在创业初期犯过错误，未能充分发挥长板优势，而把同样级别的精力投入到了短板上。他认为，应该迅速弥补短板，发挥长板优势，否则会被短板拖累。\n\n\
                        此外，SPEAKER_00还谈到了一些与他个人品牌密切相关的词语，如“彪悍”、“奋斗”、“工匠精神”和“情怀”等。他表示，这些词语是用自己的经历和价值观传递给学生的，但并非自己特别在意。\n\n\
                        最后，SPEAKER_00谈到了一些媒体对他的评价和标签，例如“贴吧诗人”和“情怀”等。他表示，这些标签并非完全符合自己，但自己会接受这些评价，因为这些词语反映了他过去的一些观点和价值观。\n\n"
                                                            
            
            questions = type_prompt + "<|user|>\n" + question + "<|assistant|>\n"
            messages = [self.messages,
                        {'role': 'user', 'content': questions}]
            response = Generation.call(
                dashscope.Generation.Models.qwen_plus,
                messages =  messages,
                api_key =self.key,
                result_format='message'
            )
            print("+++++++++++++++++++++++++")
            print(response['message'])
            if response["output"] is None:
                return ""
            ret = response["output"]["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"error reason: {e}")
            time.sleep(1)
            print("reconnecting...")
            ret = self.get_translation_answer(question)
        return ret
    
    
if __name__ == "__main__":
    model_path = "glm/chatglm3-6b-32k"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device='cuda')
    model = model.eval()
    response, _ = model.chat(tokenizer, "你好", history=[])
    print(response)