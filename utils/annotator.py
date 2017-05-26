from stanza.nlp.corenlp import CoreNLPClient
from stanza.nlp.corenlp import CoreNLP_pb2
from stanza.nlp.corenlp import AnnotatedDocument
from google.protobuf.internal.decoder import _DecodeVarint

class Annotator:
    def __init__(self, annotators=['ssplit', 'pos'], server="http://localhost:9000"):
        self.client = CoreNLPClient(server=server, default_annotators=annotators)

    def annotate_pos(self, text):
        properties = {
            'annotators': ','.join(self.client.default_annotators),
            'outputFormat': 'serialized',
            'tokenize.whitespace': 'true',
            'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer'
        }
        r = self.client._request(text, properties)
        buffer = r.content  # bytes
        size, pos = _DecodeVarint(buffer, 0)
        buffer = buffer[pos:(pos + size)]
        r.connection.close()
        doc = CoreNLP_pb2.Document()
        doc.ParseFromString(buffer)
        annotations = AnnotatedDocument.from_pb(doc)
        tokens = []
        for sentence in annotations.sentences:
            tokens += [token.pos for token in sentence.tokens]
        return tokens
