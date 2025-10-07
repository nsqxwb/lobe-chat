import { Markdown, MarkdownProps } from '@lobehub/ui';
import { createStyles } from 'antd-style';
import { memo } from 'react';
import { Flexbox } from 'react-layout-kit';

import { AssistantContentBlock } from '@/types/message';

import { normalizeThinkTags, processWithArtifact } from '../../utils/markdown';
import ImageFileListViewer from '../User/ImageFileListViewer';
import Tool from './Tool';

const LINE_LENGTH = 10;

const useStyles = createStyles(({ css, token }) => {
  const lineBlock = css`
    content: '';

    position: absolute;
    inset-inline-start: 50%;
    transform: translateX(-50%);

    width: 1px;
    height: ${LINE_LENGTH}px;

    background: ${token.colorBorderSecondary};
  `;

  return {
    blockEnd: css`
      padding-block: 0 8px;
      padding-inline: 12px;
      border: 1px solid ${token.colorBorder};
      border-block-start: 0;
      border-end-start-radius: 8px;
      border-end-end-radius: 8px;
    `,
    blockInMiddle: css`
      padding-inline: 12px;
      border: 1px solid ${token.colorBorder};
      border-block-start: 0;
      border-block-end: 0;
      border-radius: 0;
    `,
    blockStandalone: css`
      padding-block: 6px;
      padding-inline: 12px;
      border: 1px solid ${token.colorBorder};
      border-radius: 8px;
    `,
    blockStart: css`
      padding-block: 8px 0;
      padding-inline: 12px;
      border: 1px solid ${token.colorBorder};
      border-block-end: 0;
      border-start-start-radius: 8px;
      border-start-end-radius: 8px;
    `,
    toolConnectLine: css`
      position: relative;
      padding-block-end: ${LINE_LENGTH}px;

      ::after {
        inset-block-end: 0;
        ${lineBlock};
      }
    `,
  };
});

interface AssistantBlockProps extends AssistantContentBlock {
  blockPosition?: 'start' | 'middle' | 'end' | 'standalone';
  index: number;
  markdownProps?: Omit<MarkdownProps, 'className' | 'style' | 'children'>;
}
export const AssistantBlock = memo<AssistantBlockProps>(
  ({ id, tools, content, imageList, markdownProps, blockPosition }) => {
    const { styles, cx } = useStyles();
    // const generating = useChatStore(chatSelectors.isMessageGenerating(id));

    // const isToolCallGenerating = generating && (content === LOADING_FLAT || !content) && !!tools;
    const message = normalizeThinkTags(processWithArtifact(content));

    const showImageItems = !!imageList && imageList.length > 0;

    if (tools && tools.length > 0) {
      const toolsCount = tools.length;
      const hasMultipleTools = toolsCount > 1;

      return (
        <Flexbox
          className={cx(
            blockPosition === 'standalone' && styles.blockStandalone,
            blockPosition === 'start' && styles.blockStart,
            blockPosition === 'middle' && styles.blockInMiddle,
            blockPosition === 'end' && styles.blockEnd,
          )}
          gap={0}
        >
          {tools.map((toolCall, index) => {
            const isLastTool = index === toolsCount - 1;
            const showConnectLine = hasMultipleTools && !isLastTool;

            return (
              <Flexbox className={cx(showConnectLine && styles.toolConnectLine)} key={toolCall.id}>
                <Tool
                  apiName={toolCall.apiName}
                  arguments={toolCall.arguments}
                  id={toolCall.id}
                  identifier={toolCall.identifier}
                  index={index}
                  messageId={id}
                  payload={toolCall}
                  type={toolCall.type}
                />
              </Flexbox>
            );
          })}
        </Flexbox>
      );
    }

    return (
      <Flexbox gap={8} id={id}>
        <Markdown {...markdownProps} variant={'chat'}>
          {message}
        </Markdown>
        {showImageItems && <ImageFileListViewer items={imageList} />}
      </Flexbox>
    );
  },
);
