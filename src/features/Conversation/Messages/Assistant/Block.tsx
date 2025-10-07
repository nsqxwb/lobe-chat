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
    inset-inline-start: 3.5px;

    width: 1px;

    background: ${token.colorBorder};
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
    status: css`
      position: relative;
      z-index: 1;

      display: flex;
      flex-shrink: 0;
      align-items: center;
      justify-content: center;

      width: 8px;
      height: 8px;
      margin-block-start: 8px;
      border-radius: 50%;

      background-color: ${token.colorTextDescription};
    `,
    statusEnd: css`
      &::before {
        ${lineBlock};
        inset-block-start: -${LINE_LENGTH}px;
      }
    `,
    statusMiddle: css`
      &::before {
        ${lineBlock};
        inset-block-start: -${LINE_LENGTH}px;
      }

      &::after {
        ${lineBlock};
        inset-block-end: -${LINE_LENGTH}px;
      }
    `,
    statusStart: css`
      &::after {
        ${lineBlock};
        inset-block: 12px 0;
      }
    `,
    toolContent: css`
      flex: 1;
      min-width: 0;
    `,
    toolHeader: css`
      position: relative;
      display: flex;
      gap: 4px;
    `,
    toolHeaderFirst: css`
      padding-block-start: 0;
    `,
    toolHeaderLast: css`
      padding-block-end: 0;
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
            const isFirstTool = index === 0;
            const isLastTool = index === toolsCount - 1;
            const hasMultipleTools = toolsCount > 1;

            // 根据 blockPosition 判断是否需要连接到其他 block
            const needConnectToPrevBlock =
              blockPosition === 'middle' || blockPosition === 'end';
            const needConnectToNextBlock =
              blockPosition === 'middle' || blockPosition === 'start';

            let statusClassName = styles.status;
            let headerClassName = styles.toolHeader;

            // 第一个 tool
            if (isFirstTool) {
              headerClassName = cx(styles.toolHeader, styles.toolHeaderFirst);

              if (needConnectToPrevBlock) {
                // 需要连接到上一个 block
                if (hasMultipleTools || needConnectToNextBlock) {
                  // 有下一个 tool 或需要连接到下一个 block，需要上下连接线
                  statusClassName = cx(styles.status, styles.statusMiddle);
                } else {
                  // 只有一个 tool 且不需要连接到下一个 block，只需要上方连接线
                  statusClassName = cx(styles.status, styles.statusEnd);
                }
              } else {
                // 不需要连接到上一个 block
                if (hasMultipleTools) {
                  // 有下一个 tool，需要下方连接线
                  statusClassName = cx(styles.status, styles.statusStart);
                } else {
                  // 只有一个 tool，不需要任何连接线
                  statusClassName = styles.status;
                }
              }
            }
            // 最后一个 tool（且不是第一个）
            else if (isLastTool) {
              headerClassName = cx(styles.toolHeader, styles.toolHeaderLast);

              if (needConnectToNextBlock) {
                // 需要连接到下一个 block，需要上下连接线
                statusClassName = cx(styles.status, styles.statusMiddle);
              } else {
                // 不需要连接到下一个 block，只需要上方连接线
                statusClassName = cx(styles.status, styles.statusEnd);
              }
            }
            // 中间的 tool
            else {
              headerClassName = styles.toolHeader;
              statusClassName = cx(styles.status, styles.statusMiddle);
            }

            return (
              <Flexbox key={toolCall.id}>
                <div className={headerClassName}>
                  <div className={statusClassName} />
                  <div className={styles.toolContent}>
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
                  </div>
                </div>
              </Flexbox>
            );
          })}
        </Flexbox>
      );
    }

    return (
      <Flexbox gap={8} id={id} style={{ marginBlock: 8 }}>
        <Markdown {...markdownProps} variant={'chat'}>
          {message}
        </Markdown>
        {showImageItems && <ImageFileListViewer items={imageList} />}
      </Flexbox>
    );
  },
);
