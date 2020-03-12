#include "marian.h"

#include "models/model_factory.h"
#include "models/encoder_decoder.h"
#include "models/encoder_classifier.h"
#include "models/bert.h"

#include "models/costs.h"

#include "models/amun.h"
#include "models/nematus.h"
#include "models/s2s.h"
#include "models/transformer_factory.h"

#ifdef CUDNN
#include "models/char_s2s.h"
#endif

#ifdef COMPILE_EXAMPLES
#include "examples/mnist/model.h"
#ifdef CUDNN
#include "examples/mnist/model_lenet.h"
#endif
#endif

namespace marian {
namespace models {

Ptr<EncoderBase> EncoderFactory::construct(Ptr<ExpressionGraph> graph) {
  if(options_->get<std::string>("type") == "s2s")
    return New<EncoderS2S>(graph, options_);

#ifdef CUDNN
  if(options_->get<std::string>("type") == "char-s2s")
    return New<CharS2SEncoder>(graph, options_);
#endif

  if(options_->get<std::string>("type") == "transformer")
    return NewEncoderTransformer(graph, options_);

  if(options_->get<std::string>("type") == "bert-encoder")
    return New<BertEncoder>(graph, options_);

  ABORT("Unknown encoder type");
}

Ptr<DecoderBase> DecoderFactory::construct(Ptr<ExpressionGraph> graph) {
  if(options_->get<std::string>("type") == "s2s")
    return New<DecoderS2S>(graph, options_);
  if(options_->get<std::string>("type") == "transformer")
    return NewDecoderTransformer(graph, options_);
  ABORT("Unknown decoder type");
}

Ptr<ClassifierBase> ClassifierFactory::construct(Ptr<ExpressionGraph> graph) {
  if(options_->get<std::string>("type") == "bert-masked-lm")
    return New<BertMaskedLM>(graph, options_);
  else if(options_->get<std::string>("type") == "bert-classifier")
    return New<BertClassifier>(graph, options_);
  else
    ABORT("Unknown classifier type");
}

Ptr<IModel> EncoderDecoderFactory::construct(Ptr<ExpressionGraph> graph) {
  Ptr<EncoderDecoder> encdec;
  if(options_->get<std::string>("type") == "amun")
    encdec = New<Amun>(graph, options_);
  else if(options_->get<std::string>("type") == "nematus")
    encdec = New<Nematus>(graph, options_);
  else
    encdec = New<EncoderDecoder>(graph, options_);

  for(auto& ef : encoders_)
    encdec->push_back(ef(options_).construct(graph));

  for(auto& df : decoders_)
    encdec->push_back(df(options_).construct(graph));

  return encdec;
}

Ptr<IModel> EncoderClassifierFactory::construct(Ptr<ExpressionGraph> graph) {
  Ptr<EncoderClassifier> enccls;
  if(options_->get<std::string>("type") == "bert")
    enccls = New<BertEncoderClassifier>(options_);
  else if(options_->get<std::string>("type") == "bert-classifier")
    enccls = New<BertEncoderClassifier>(options_);
  else
    enccls = New<EncoderClassifier>(options_);

  for(auto& ef : encoders_)
    enccls->push_back(ef(options_).construct(graph));

  for(auto& cf : classifiers_)
    enccls->push_back(cf(options_).construct(graph));

  return enccls;
}

Ptr<IModel> createBaseModelByType(std::string type, usage use, Ptr<Options> options) {
  Ptr<ExpressionGraph> graph = nullptr; // graph unknown at this stage
  // clang-format off
  if(type == "s2s" || type == "amun" || type == "nematus") {
    return models::encoder_decoder(options->with(
         "usage", use,
         "original-type", type))
        .push_back(models::encoder()("type", "s2s"))
        .push_back(models::decoder()("type", "s2s"))
        .construct(graph);
  }

  else if(type == "transformer") {
#if 1
    auto newOptions = options->with("usage", use);
    auto res = New<EncoderDecoder>(graph, newOptions);
    res->push_back(New<EncoderTransformer>(graph, newOptions->with("type", "transformer")));
    res->push_back(New<DecoderTransformer>(graph, newOptions->with("type", "transformer")));
    return res;
#else
    return models::encoder_decoder(options->with(
         "usage", use))
        .push_back(models::encoder()("type", "transformer"))
        .push_back(models::decoder()("type", "transformer"))
        .construct(graph);
#endif
  }

  else if(type == "transformer_s2s") {
    return models::encoder_decoder()(options)
        ("usage", use)
        ("original-type", type)
        .push_back(models::encoder()("type", "transformer"))
        .push_back(models::decoder()("type", "s2s"))
        .construct(graph);
  }

  else if(type == "lm") {
    auto idx = options->has("index") ? options->get<size_t>("index") : 0;
    std::vector<int> dimVocabs = options->get<std::vector<int>>("dim-vocabs");
    int vocab = dimVocabs[0];
    dimVocabs.resize(idx + 1);
    std::fill(dimVocabs.begin(), dimVocabs.end(), vocab);

    return models::encoder_decoder(options->with(
         "usage", use,
         "type", "s2s",
         "original-type", type))
        .push_back(models::decoder()
                   ("index", idx)
                   ("dim-vocabs", dimVocabs))
        .construct(graph);
  }

  else if(type == "multi-s2s") {
    size_t numEncoders = 2;
    auto ms2sFactory = models::encoder_decoder()(options)
        ("usage", use)
        ("type", "s2s")
        ("original-type", type);

    for(size_t i = 0; i < numEncoders; ++i) {
      auto prefix = "encoder" + std::to_string(i + 1);
      ms2sFactory.push_back(models::encoder()("prefix", prefix)("index", i));
    }

    ms2sFactory.push_back(models::decoder()("index", numEncoders));

    return ms2sFactory.construct(graph);
  }

  else if(type == "shared-multi-s2s") {
    size_t numEncoders = 2;
    auto ms2sFactory = models::encoder_decoder()(options)
        ("usage", use)
        ("type", "s2s")
        ("original-type", type);

    for(size_t i = 0; i < numEncoders; ++i) {
      auto prefix = "encoder";
      ms2sFactory.push_back(models::encoder()("prefix", prefix)("index", i));
    }

    ms2sFactory.push_back(models::decoder()("index", numEncoders));

    return ms2sFactory.construct(graph);
  }

  else if(type == "multi-transformer") {
    size_t numEncoders = 2;
    auto mtransFactory = models::encoder_decoder()(options)
        ("usage", use)
        ("type", "transformer")
        ("original-type", type);

    for(size_t i = 0; i < numEncoders; ++i) {
      auto prefix = "encoder" + std::to_string(i + 1);
      mtransFactory.push_back(models::encoder()("prefix", prefix)("index", i));
    }
    mtransFactory.push_back(models::decoder()("index", numEncoders));

    return mtransFactory.construct(graph);
  }

  else if(type == "shared-multi-transformer") {
    size_t numEncoders = 2;
    auto mtransFactory = models::encoder_decoder()(options)
        ("usage", use)
        ("type", "transformer")
        ("original-type", type);

    for(size_t i = 0; i < numEncoders; ++i) {
      auto prefix = "encoder";
      mtransFactory.push_back(models::encoder()("prefix", prefix)("index", i));
    }
    mtransFactory.push_back(models::decoder()("index", numEncoders));

    return mtransFactory.construct(graph);
  }

  else if(type == "lm-transformer") {
    auto idx = options->has("index") ? options->get<size_t>("index") : 0;
    std::vector<int> dimVocabs = options->get<std::vector<int>>("dim-vocabs");
    int vocab = dimVocabs[0];
    dimVocabs.resize(idx + 1);
    std::fill(dimVocabs.begin(), dimVocabs.end(), vocab);

    return models::encoder_decoder()(options)
        ("usage", use)
        ("type", "transformer")
        ("original-type", type)
        .push_back(models::decoder()
                   ("index", idx)
                   ("dim-vocabs", dimVocabs))
        .construct(graph);
  }

  else if(type == "bert") {                      // for full BERT training
    return models::encoder_classifier()(options) //
        ("original-type", "bert")                // so we can query this
        ("usage", use)                           //
        .push_back(models::encoder()             //
                   ("type", "bert-encoder")      // close to original transformer encoder
                   ("index", 0))                 //
        .push_back(models::classifier()          //
                   ("prefix", "masked-lm")       // prefix for parameter names
                   ("type", "bert-masked-lm")    //
                   ("index", 0))                 // multi-task learning with MaskedLM
        .push_back(models::classifier()          //
                   ("prefix", "next-sentence")   // prefix for parameter names
                   ("type", "bert-classifier")   //
                   ("index", 1))                 // next sentence prediction
        .construct(graph);
  }

  else if(type == "bert-classifier") {           // for BERT fine-tuning on non-BERT classification task
    return models::encoder_classifier()(options) //
        ("original-type", "bert-classifier")     // so we can query this if needed
        ("usage", use)                           //
        .push_back(models::encoder()             //
                   ("type", "bert-encoder")      //
                   ("index", 0))                 // close to original transformer encoder
        .push_back(models::classifier()          //
                   ("type", "bert-classifier")   //
                   ("index", 1))                 // next sentence prediction
        .construct(graph);
  }

  else if(type == "bert-tagger") {
    auto bertOptions = options->with("usage", use, "original-type", "bert-tagger");
    auto encoderOptions = bertOptions->with("type", "bert-encoder", "index", 0);
    auto classifierOptions = bertOptions->with("type", "bert-tagger", "index", 1);
    auto model = New<BertEncoderClassifier>(bertOptions);
    model->push_back(New<BertEncoder>(graph, encoderOptions));
    model->push_back(New<BertTagger>(graph, classifierOptions));
    return model;
  }

#ifdef COMPILE_EXAMPLES
  else if(type == "mnist-ffnn")
    return New<MnistFeedForwardNet>(options);
#endif
#ifdef CUDNN
#ifdef COMPILE_EXAMPLES
  else if(type == "mnist-lenet")
    return New<MnistLeNet>(options);
#endif
  else if(type == "char-s2s") {
    return models::encoder_decoder()(options)
        ("usage", use)
        ("original-type", type)
        .push_back(models::encoder()("type", "char-s2s"))
        .push_back(models::decoder()("type", "s2s"))
        .construct(graph);
  }
#endif

  // clang-format on
  else
    ABORT("Unknown model type: {}", type);
}

Ptr<IModel> createModelFromOptions(Ptr<Options> options, usage use) {
  std::string type = options->get<std::string>("type");
  auto baseModel = createBaseModelByType(type, use, options);

  // add (log)softmax if requested
  if (use == usage::translation) {
    if(std::dynamic_pointer_cast<EncoderDecoder>(baseModel)) {
      if(options->get<bool>("output-sampling", false))
        return New<Stepwise>(std::dynamic_pointer_cast<EncoderDecoder>(baseModel), New<GumbelSoftmaxStep>());
      else
        return New<Stepwise>(std::dynamic_pointer_cast<EncoderDecoder>(baseModel), New<LogSoftmaxStep>());
    }
#ifdef COMPILE_EXAMPLES
    // note: 'usage::translation' here means 'inference'
    else if (std::dynamic_pointer_cast<MnistFeedForwardNet>(baseModel))
      return New<Scorer>(baseModel, New<MNISTLogsoftmax>());
#ifdef CUDNN
    else if (std::dynamic_pointer_cast<MnistLeNet>(baseModel))
      return New<Scorer>(baseModel, New<MNISTLogsoftmax>());
#endif
#endif
    else
      ABORT("'usage' parameter 'translation' cannot be applied to model type: {}", type);
  }
  else if (use == usage::raw)
    return baseModel;
  else
    ABORT("'Usage' parameter must be 'translation' or 'raw'");
}

Ptr<ICriterionFunction> createCriterionFunctionFromOptions(Ptr<Options> options, usage use) {
  std::string type = options->get<std::string>("type");
  auto baseModel = createBaseModelByType(type, use, options);

  // add cost function
  ABORT_IF(use != usage::training && use != usage::scoring, "'Usage' parameter must be 'training' or 'scoring'");
  // note: usage::scoring means "score the loss function", hence it uses a Trainer (not Scorer, which is for decoding)
  // @TODO: Should we define a new class that does not compute gradients?
  if (std::dynamic_pointer_cast<EncoderDecoder>(baseModel))
    return New<Trainer>(baseModel, New<EncoderDecoderCECost>(options));
  else if (std::dynamic_pointer_cast<EncoderClassifier>(baseModel))
    return New<Trainer>(baseModel, New<EncoderClassifierCECost>(options));
#ifdef COMPILE_EXAMPLES
  // @TODO: examples should be compiled optionally
  else if (std::dynamic_pointer_cast<MnistFeedForwardNet>(baseModel))
    return New<Trainer>(baseModel, New<MNISTCrossEntropyCost>());
#ifdef CUDNN
  else if (std::dynamic_pointer_cast<MnistLeNet>(baseModel))
    return New<Trainer>(baseModel, New<MNISTCrossEntropyCost>());
#endif
#endif
  else
    ABORT("Criterion function unknown for model type: {}", type);
}

}  // namespace models
}  // namespace marian
